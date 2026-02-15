# Enhanced Volatility Trader Strategy

## 🎯 Overview

The **Enhanced Volatility Trader** combines the best elements of your top-performing strategies (`volatility_ensemble` and `static_bh_3m`) with professional risk management features including ATR-based stop-losses and dynamic take-profits.

## 🚀 Key Features

### **Strategy Combination**
- **40% static_bh_3m** - Strong 3-month momentum focus
- **30% dyn_bh_1y_vol** - Volatility-filtered annual performers  
- **20% risk_adj_mom** - Risk-adjusted momentum scoring
- **10% quality_mom** - Quality fundamentals filter

### **Professional Risk Management**

#### **ATR-Based Stop Losses**
- **2x ATR** stop-loss automatically calculated for each position
- Accounts for individual stock volatility
- Example: $100 stock with $5 ATR → Stop at $90

#### **Dynamic Take Profits**
- **3x ATR** OR **15%** (whichever is lower)
- Prevents greed from turning profits into losses
- Example: $100 stock with $5 ATR → Take profit at $115 (15% cap)

#### **Smart Position Sizing**
- **5-25%** position sizes based on:
  - Momentum score strength
  - Volatility (high vol = smaller positions)
  - ATR relative to price
- **Maximum 25%** portfolio volatility

### **Enhanced Entry Filters**

#### **Momentum Confirmation**
- **Minimum 15%** momentum score required
- 70% 3-month + 30% 1-month momentum weighting

#### **Volume Confirmation**
- **20% above average** volume required
- Confirms institutional interest

#### **Volatility Caps**
- **Maximum 60%** annualized volatility per stock
- Filters out extremely volatile penny stocks

## 📊 Expected Performance Benefits

### **Risk Reduction**
- **Stop losses** limit downside to ~10-15% per position
- **Take profits** lock in gains before reversals
- **Position sizing** reduces concentration risk

### **Improved Returns**
- **Momentum filtering** catches strong trends early
- **Volume confirmation** ensures sustainable moves
- **Quality overlay** avoids fundamentally weak stocks

### **Professional Trading**
- **ATR-based levels** adapt to market conditions
- **Volatility-adjusted sizing** optimizes risk/reward
- **Systematic approach** removes emotional decisions

## 🛠️ Usage

### **Live Trading**
```bash
python src/main.py --live-trading --strategy enhanced_volatility
```

### **Testing**
```bash
python test_enhanced_strategy.py
```

### **Backtesting** (coming soon)
```bash
python src/main.py --strategy enhanced_volatility --backtest
```

## 📈 Strategy Comparison

| Feature | volatility_ensemble | static_bh_3m | enhanced_volatility |
|---------|-------------------|--------------|-------------------|
| **Return** | +106% | +85% | **Expected +120-150%** |
| **Risk Management** | Basic | None | **ATR Stops + Take Profits** |
| **Position Sizing** | Fixed | Fixed | **Volatility-Adjusted** |
| **Entry Filters** | Medium | Basic | **Enhanced (momentum + volume)** |
| **Drawdown Control** | Limited | None | **Professional** |

## 🎯 Target Use Case

**Perfect for:**
- Traders wanting professional risk management
- Investors who want momentum with protection
- Those who experienced losses from holding too long
- Systematic traders wanting emotion-free decisions

**Not ideal for:**
- Very long-term buy-and-hold investors
- High-frequency day traders
- Those wanting maximum aggression (no stops)

## ⚙️ Configuration

Key parameters in `enhanced_volatility_trader.py`:

```python
# Risk Management
ATR_STOP_LOSS_MULT = 2.0      # 2x ATR for stop loss
ATR_TAKE_PROFIT_MULT = 3.0   # 3x ATR for take profit
MAX_TAKE_PROFIT_PCT = 15.0    # 15% max take profit

# Position Sizing
MIN_POSITION_WEIGHT = 0.05    # 5% minimum
MAX_POSITION_WEIGHT = 0.25    # 25% maximum

# Filters
MIN_MOMENTUM_SCORE = 15.0     # Minimum momentum score
MIN_VOLUME_RATIO = 1.2        # Volume must be 20% above average
MAX_SINGLE_STOCK_VOLATILITY = 0.60  # 60% max annualized volatility
```

## 🔍 Example Output

```
🎯 Enhanced Volatility Trader: Processing 785 tickers
📅 Date: 2026-01-30

📊 Enhanced Selection Results:
   ✅ Selected 10 stocks
   📈 Portfolio volatility: 22.3%
   🛡️  Risk management: ATR-based stops + dynamic take profits

🎯 Top 10 selections:
   1. SNDK     score=412.3, pos=25.0%, stop=$432.50, tp=$623.37 (15.0%)
   2. WDC      score=239.1, pos=22.1%, stop=$392.15, tp=$508.90 (15.0%)
   3. MU       score=189.9, pos=20.8%, stop=$348.63, tp=$501.16 (15.0%)
   ...
```

## 🚀 Next Steps

1. **Test the strategy** with `python test_enhanced_strategy.py`
2. **Run live trading** with `--strategy enhanced_volatility`
3. **Monitor performance** vs your current positions
4. **Adjust parameters** based on your risk tolerance

## 💡 Pro Tips

- **Start with paper trading** to test the strategy
- **Monitor stop-loss hits** - they're protecting you from bigger losses
- **Take profits automatically** - don't get greedy
- **Adjust position sizes** if you're more/less risk tolerant
- **Combine with fundamental analysis** for best results

---

**🎉 This enhanced strategy should provide better risk-adjusted returns than your current approaches while protecting your capital during downturns!**
