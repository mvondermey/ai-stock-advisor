# AI Strategy Improvement Plan

## 🚀 IMPROVEMENTS IMPLEMENTED

### Phase 1: Critical Infrastructure Fixes ✅
- **✅ ENABLED AI TRAINING**: `ENABLE_1YEAR_TRAINING = True` (was False)
- **✅ FIXED TRANSACTION COST GUARD**: Corrected function parameter order
- **✅ IMPROVED AI PORTFOLIO TRAINING**: Reduced evaluation window from 60→30 days

### Phase 2: Model Architecture Improvements ✅
- **✅ INCREASED SEQUENCE LENGTH**: 60→120 days (better pattern recognition)
- **✅ ENHANCED LSTM ARCHITECTURE**:
  - Hidden size: 64→128 neurons
  - Layers: 2→3 layers (deeper network)
  - Dropout: 0.2→0.3 (better regularization)
  - Epochs: 50→100 (better training)
  - Batch size: 64→32 (better convergence)
  - Learning rate: 0.001→0.0005 (more stable)

### Phase 3: Trading Logic Improvements ✅
- **✅ MAINTAIN DAILY REBALANCING**: Keep 1-day frequency for optimal responsiveness
- **✅ LOWER AI PORTFOLIO THRESHOLD**: 50%→30% annual (more trading opportunities)
- **✅ REMOVED TARGET_PERCENTAGE**: Cleaned up unused parameter (regression mode uses actual returns)

## EXPECTED PERFORMANCE IMPROVEMENTS

### Before vs After:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Sequence Length** | 60 days | 120 days | +100% |
| **Model Capacity** | 64 neurons | 128 neurons | +100% |
| **Training Depth** | 2 layers | 3 layers | +50% |
| **Training Time** | 50 epochs | 100 epochs | +100% |
| **Rebalancing** | Daily | Daily | Maintained for optimal responsiveness |

### Performance Expectations:
- **Better Pattern Recognition**: 120-day sequences capture longer trends
- **Smarter Models**: Deeper LSTM networks learn complex relationships
- **Daily Responsiveness**: Maintain daily rebalancing for optimal market responsiveness
- **More Training**: 100 epochs ensure better model convergence

### How AI Strategy Actually Works:

1. **Prediction Phase:**
   ```python
   # Each day, AI models predict forward returns for all candidates
   predictions = [(ticker, predicted_return), ...]
   ```

2. **Selection Phase:**
   ```python
   # Select top 10 tickers by highest predicted returns
   predictions.sort(key=lambda x: x[1], reverse=True)
   selected_stocks = [ticker for ticker, _ in predictions[:PORTFOLIO_SIZE]]  # PORTFOLIO_SIZE = 10
   ```

3. **Rebalancing Phase:**
   ```python
   # Only rebalance if:
   # 1. Portfolio changed (different stocks) AND
   # 2. Transaction cost guard passed (portfolio grew enough to cover costs)
   should_rebal, reason = _should_rebalance_by_profit_since_last_rebalance(...)
   ```

### The Real Impact of TARGET_PERCENTAGE Removal:

**TARGET_PERCENTAGE has been completely removed** from the codebase since it was only used for legacy classification mode. The current system uses **regression mode** where:

- **Models predict exact returns** (e.g., 2.3%, -1.5%, 3.7%)
- **Top 10 stocks selected by highest predictions**
- **Transaction cost guard prevents excessive trading**
- **No arbitrary target thresholds** - cleaner, more accurate system

## TECHNICAL IMPROVEMENTS

### 1. Model Architecture
```python
# Before: Simple LSTM
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
SEQUENCE_LENGTH = 60

# After: Enhanced LSTM
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 3
SEQUENCE_LENGTH = 120
```

### 2. Trading Logic
```python
# Before: Had unused TARGET_PERCENTAGE
TARGET_PERCENTAGE = 0.006  # Not used in regression mode
AI_REBALANCE_FREQUENCY_DAYS = 1  # Daily

# After: Clean regression mode
# TARGET_PERCENTAGE removed - models predict actual returns
AI_REBALANCE_FREQUENCY_DAYS = 1  # Daily (maintained for responsiveness)
```

### 3. Training Parameters
```python
# Before: Basic training
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 64
LSTM_LEARNING_RATE = 0.001

# After: Enhanced training
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 32
LSTM_LEARNING_RATE = 0.0005
```

## 🚨 NEXT STEPS

### Immediate Actions:
1. **🧪 TEST THE CHANGES**: Run backtest with new parameters
2. **📊 MONITOR PERFORMANCE**: Compare with previous results
3. **🔧 FINE-TUNE**: Adjust parameters based on results

### Future Enhancements:
1. **🎯 ENSEMBLE MODELS**: Combine LSTM + XGBoost predictions
2. **📱 MARKET REGIMES**: Add volatility-based model switching
3. **🔄 ONLINE LEARNING**: Update models incrementally
4. **📊 FEATURE ENGINEERING**: Add more technical indicators

## ⚡ QUICK TEST COMMAND

```bash
# Test the improved AI strategy
python src/main.py --live-trading --strategy ai_individual

# Or run full backtest
python src/main.py
```

## 📈 SUCCESS METRICS

### What to Watch For:
- **✅ Higher Hit Rate**: More accurate predictions
- **✅ Better Returns**: Improved portfolio performance
- **✅ Lower Drawdowns**: Better risk management
- **✅ Fewer Errors**: No more transaction cost guard failures
- **✅ Stable Training**: No more "too few training samples" errors

### Target Improvements:
- **🎯 Hit Rate**: 50% → 60%+
- **🎯 Annual Return**: 10% → 20%+
- **🎯 Max Drawdown**: -20% → -15%
- **🎯 Sharpe Ratio**: 0.5 → 1.0+

---

**🚀 READY TO TEST! All improvements implemented and ready for backtesting.**
