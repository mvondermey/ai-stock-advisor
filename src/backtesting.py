import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Import from config, ml_models, feature_engineering
from config import (
    ATR_PERIOD, INVESTMENT_PER_STOCK, TRANSACTION_COST, MIN_PROBA_BUY,
    MIN_PROBA_SELL, USE_MODEL_GATE, USE_SIMPLE_RULE_STRATEGY,
    SIMPLE_RULE_TRAILING_STOP_PERCENT, SIMPLE_RULE_TAKE_PROFIT_PERCENT
)
from ml_models import LSTMClassifier, GRUClassifier, PYTORCH_AVAILABLE, CUDA_AVAILABLE
from feature_engineering import fetch_training_data

class RuleTradingEnv:
    """SMA cross + ATR trailing stop/TP + risk-based sizing. Optional ML gate to allow buys."""
    def __init__(self, df: pd.DataFrame, ticker: str, initial_balance: float, transaction_cost: float,
                 model_buy=None, model_sell=None, scaler=None, min_proba_buy: float = MIN_PROBA_BUY, min_proba_sell: float = MIN_PROBA_SELL, use_gate: bool = USE_MODEL_GATE,
                 feature_set: Optional[List[str]] = None,
                 per_ticker_min_proba_buy: Optional[float] = None, per_ticker_min_proba_sell: Optional[float] = None,
                 use_simple_rule_strategy: bool = USE_SIMPLE_RULE_STRATEGY,
                 simple_rule_trailing_stop_percent: float = SIMPLE_RULE_TRAILING_STOP_PERCENT,
                 simple_rule_take_profit_percent: float = SIMPLE_RULE_TAKE_PROFIT_PERCENT):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")
        self.df = df.reset_index()
        self.ticker = ticker
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.model_buy = model_buy
        self.model_sell = model_sell
        self.scaler = scaler
        self.min_proba_buy = float(per_ticker_min_proba_buy if per_ticker_min_proba_buy is not None else min_proba_buy)
        self.min_proba_sell = float(per_ticker_min_proba_sell if per_ticker_min_proba_sell is not None else min_proba_sell)
        self.use_gate = bool(use_gate) and (scaler is not None)
        self.use_simple_rule_strategy = use_simple_rule_strategy
        self.simple_rule_trailing_stop_percent = simple_rule_trailing_stop_percent
        self.simple_rule_take_profit_percent = simple_rule_take_profit_percent
        
        df_with_features, actual_feature_set_for_env = fetch_training_data(
            ticker, df.copy(), target_percentage=0.0, class_horizon=1
        )
        self.df = df_with_features.reset_index()
        self.feature_set = actual_feature_set_for_env
        
        self.reset()
        self._prepare_data()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = 0.0
        self.entry_price: Optional[float] = None
        self.highest_since_entry: Optional[float] = None
        self.entry_atr: Optional[float] = None
        self.holding_bars = 0
        self.portfolio_history: List[float] = [self.initial_balance]
        self.trade_log: List[Tuple] = []
        self.last_ai_action: str = "HOLD"
        self.last_buy_prob: float = 0.0
        self.last_sell_prob: float = 0.0
        self.trailing_stop_price: Optional[float] = None
        self.take_profit_price: Optional[float] = None
        
    def _prepare_data(self):
        if self.df.empty:
            print(f"  [DIAGNOSTIC] {self.ticker}: DataFrame is empty in _prepare_data. Skipping further prep.")
            return

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        if "Volume" not in self.df.columns: self.df["Volume"] = 0
        if "High" not in self.df.columns: self.df["High"] = self.df["Close"]
        if "Low" not in self.df.columns: self.df["Low"] = self.df["Close"]
        if "Open" not in self.df.columns: self.df["Open"] = self.df["Close"]
            
        self.df = self.df.dropna(subset=["Close"])
        if self.df.empty:
            print(f"  [DIAGNOSTIC] {self.ticker}: DataFrame became empty after dropping NaNs in 'Close' during _prepare_data. Skipping further prep.")
            return
        self.df = self.df.reset_index(drop=True)
        self.df = self.df.ffill().bfill()
        
        close = self.df["Close"]
        high = self.df["High"] if "High" in self.df.columns else None
        low  = self.df["Low"]  if "Low" in self.df.columns else None
        prev_close = close.shift(1)
        if high is not None and low is not None:
            hl = (high - low).abs()
            h_pc = (high - prev_close).abs()
            l_pc = (low  - prev_close).abs()
            tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
            self.df["ATR"] = tr.rolling(ATR_PERIOD).mean()
        else:
            ret = close.pct_change(fill_method=None)
            self.df["ATR"] = (ret.rolling(ATR_PERIOD).std() * close).rolling(2).mean()
        
        self.df['ATR_MED'] = self.df['ATR'].rolling(50).median()

        first_valid_atr_idx = self.df['ATR'].first_valid_index()
        if first_valid_atr_idx is not None:
            self.current_step = self.df.index.get_loc(first_valid_atr_idx)
        else:
            pass

    def _date_at(self, i: int) -> str:
        if "Date" in self.df.columns:
            return str(self.df.loc[i, "Date"])
        return str(i)

    def _get_model_prediction(self, i: int, model) -> float:
        """Helper to get a single model's prediction probability."""
        if not self.use_gate or model is None:
            return 0.0
        row = self.df.loc[i]
        
        model_feature_names = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else self.feature_set
        
        feature_values = {f: row.get(f, 0.0) for f in model_feature_names}
        
        X_df = pd.DataFrame([feature_values], columns=model_feature_names)
        
        X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        if X_df.isnull().all().any():
            print(f"  [{self.ticker}] Critical: Feature column is all NaN after fillna at step {i}. Skipping prediction.")
            return 0.0

        try:
            if PYTORCH_AVAILABLE and isinstance(model, (LSTMClassifier, GRUClassifier)):
                # For PyTorch models, we need to scale the data and create sequences
                # The scaler for DL models is MinMaxScaler, which was fitted on unsequenced data
                # We need to get the last SEQUENCE_LENGTH rows for prediction
                # SEQUENCE_LENGTH is a global constant, need to import it
                from config import SEQUENCE_LENGTH
                start_idx = max(0, i - SEQUENCE_LENGTH + 1)
                end_idx = i + 1
                
                if end_idx < SEQUENCE_LENGTH:
                    return 0.0
                
                historical_data_for_seq = self.df.loc[start_idx:end_idx-1, model_feature_names].copy()
                
                for col in historical_data_for_seq.columns:
                    historical_data_for_seq[col] = pd.to_numeric(historical_data_for_seq[col], errors='coerce').fillna(0.0)

                X_scaled_seq = self.scaler.transform(historical_data_for_seq)
                
                X_tensor = torch.tensor(X_scaled_seq, dtype=torch.float32).unsqueeze(0)
                
                device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
                X_tensor = X_tensor.to(device)

                model.eval()
                with torch.no_grad():
                    output = model(X_tensor)
                    return float(torch.sigmoid(output).cpu().numpy()[0][0])
            else:
                X_scaled_np = self.scaler.transform(X_df)
                X = pd.DataFrame(X_scaled_np, columns=model_feature_names)
                return float(model.predict_proba(X)[0][1])
        except Exception as e:
            print(f"  [{self.ticker}] Error in model prediction at step {i}: {e}")
            return 0.0

    def _allow_buy_by_model(self, i: int) -> bool:
        self.last_buy_prob = self._get_model_prediction(i, self.model_buy)
        return self.last_buy_prob >= self.min_proba_buy

    def _allow_sell_by_model(self, i: int) -> bool:
        self.last_sell_prob = self._get_model_prediction(i, self.model_sell)
        return self.last_sell_prob >= self.min_proba_sell

    def _position_size_from_atr(self, price: float, atr: float) -> int:
        if atr is None or np.isnan(atr) or atr <= 0 or price <= 0:
            return 0
        investment_amount = INVESTMENT_PER_STOCK
        
        qty = int(investment_amount / price)
        
        return max(qty, 0)

    def _buy(self, price: float, atr: Optional[float], date: str):
        if self.cash <= 0:
            return

        qty = self._position_size_from_atr(price, atr if atr is not None else np.nan)
        if qty <= 0:
            return

        cost = price * qty * (1 + self.transaction_cost)
        if cost > self.cash:
            qty = int(self.cash / (price * (1 + self.transaction_cost)))
        
        if qty <= 0:
            return

        fee = price * qty * self.transaction_cost
        cost = price * qty + fee

        self.cash -= cost
        self.shares += qty
        self.entry_price = price
        self.entry_atr = atr if atr is not None and not np.isnan(atr) else None
        self.highest_since_entry = price
        self.holding_bars = 0
        self.trade_log.append((date, "BUY", price, qty, self.ticker, {"fee": fee}, fee))
        self.last_ai_action = "BUY"

    def _sell(self, price: float, date: str):
        if self.shares <= 0:
            return
        qty = int(self.shares)
        proceeds = price * qty
        fee = proceeds * self.transaction_cost
        self.cash += proceeds - fee
        self.shares -= qty
        self.entry_price = None
        self.entry_atr = None
        self.highest_since_entry = None
        self.holding_bars = 0
        self.trade_log.append((date, "SELL", price, qty, self.ticker, {"fee": fee}, fee))
        self.last_ai_action = "SELL"

    def step(self):
        if self.current_step < 1:
            self.current_step += 1
            self.portfolio_history.append(self.initial_balance)
            return False

        if self.current_step >= len(self.df):
            return True

        row = self.df.iloc[self.current_step]
        
        price = float(row["Close"])
        date = self._date_at(self.current_step)
        atr = float(row.get("ATR", np.nan)) if pd.notna(row.get("ATR", np.nan)) else None

        ai_signal = False
        if not self.use_simple_rule_strategy:
            ai_signal = self._allow_buy_by_model(self.current_step)
        
        simple_rule_entry_signal = False
        if self.use_simple_rule_strategy:
            sma_short = self.df.loc[self.current_step, "SMA_F_S"]
            sma_long = self.df.loc[self.current_step, "SMA_F_L"]
            prev_sma_short = self.df.loc[self.current_step - 1, "SMA_F_S"]
            prev_sma_long = self.df.loc[self.current_step - 1, "SMA_F_L"]
            if prev_sma_short <= prev_sma_long and sma_short > sma_long:
                simple_rule_entry_signal = True

        if self.shares == 0 and (ai_signal or simple_rule_entry_signal):
            self._buy(price, atr, date)
        
        ai_exit_signal = False
        if not self.use_simple_rule_strategy:
            ai_exit_signal = self._allow_sell_by_model(self.current_step)

        simple_rule_exit_signal = False
        if self.shares > 0 and self.use_simple_rule_strategy:
            if self.highest_since_entry is None or price > self.highest_since_entry:
                self.highest_since_entry = price
            
            if self.entry_price is not None:
                self.trailing_stop_price = self.highest_since_entry * (1 - self.simple_rule_trailing_stop_percent)
                self.take_profit_price = self.entry_price * (1 + self.simple_rule_take_profit_percent)
            
            if self.trailing_stop_price is not None and price <= self.trailing_stop_price:
                simple_rule_exit_signal = True
                self.last_ai_action = "SELL (Trailing Stop)"
            elif self.take_profit_price is not None and price >= self.take_profit_price:
                simple_rule_exit_signal = True
                self.last_ai_action = "SELL (Take Profit)"

        if self.shares > 0 and (ai_exit_signal or simple_rule_exit_signal):
            self._sell(price, date)
        else:
            self.last_ai_action = "HOLD"

        port_val = self.cash + self.shares * price
        self.portfolio_history.append(port_val)
        self.current_step += 1
        return self.current_step >= len(self.df)

    def run(self) -> Tuple[float, List[Tuple], str, float, float, float]:
        if self.df.empty:
            return self.initial_balance, [], "N/A", np.nan, np.nan, 0.0
        done = False
        while not done:
            done = self.step()
        
        shares_before_liquidation = self.shares
        
        if self.shares > 0 and not self.df.empty:
            last_price = float(self.df.iloc[-1]["Close"])
            self._sell(last_price, self._date_at(len(self.df)-1))
            self.portfolio_history[-1] = self.cash
        return self.portfolio_history[-1], self.trade_log, self.last_ai_action, self.last_buy_prob, self.last_sell_prob, shares_before_liquidation

def backtest_worker(params: Tuple) -> Optional[Dict]:
    """Worker function for parallel backtesting."""
    ticker, df_backtest, capital_per_stock, model_buy, model_sell, scaler, \
        feature_set, min_proba_buy, min_proba_sell, target_percentage, \
        top_performers_data, use_simple_rule_strategy = params
    
    with open("logs/worker_debug.log", "a") as f:
        f.write(f"Worker started for ticker: {ticker}\n")

    if df_backtest.empty:
        print(f"  ⚠️ Skipping backtest for {ticker}: DataFrame is empty.")
        return None
        
    try:
        env = RuleTradingEnv(
            df=df_backtest.copy(),
            ticker=ticker,
            initial_balance=capital_per_stock,
            transaction_cost=TRANSACTION_COST,
            model_buy=model_buy,
            model_sell=model_sell,
            scaler=scaler,
            use_gate=USE_MODEL_GATE,
            feature_set=feature_set,
            per_ticker_min_proba_buy=min_proba_buy,
            per_ticker_min_proba_sell=min_proba_sell,
            use_simple_rule_strategy=use_simple_rule_strategy
        )
        final_val, trade_log, last_ai_action, last_buy_prob, last_sell_prob, shares_before_liquidation = env.run()

        start_price_bh = float(df_backtest["Close"].iloc[0])
        end_price_bh = float(df_backtest["Close"].iloc[-1])
        individual_bh_return = ((end_price_bh - start_price_bh) / start_price_bh) * 100 if start_price_bh > 0 else 0.0
        
        # Placeholder for analyze_performance, will be moved to analytics.py
        # For now, return minimal perf_data
        perf_data = {"trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

        bh_history_for_ticker = []
        if not df_backtest.empty:
            start_price = float(df_backtest["Close"].iloc[0])
            shares_bh = int(capital_per_stock / start_price) if start_price > 0 else 0
            cash_bh = capital_per_stock - shares_bh * start_price
            for price_day in df_backtest["Close"].tolist():
                bh_history_for_ticker.append(cash_bh + shares_bh * price_day)
        else:
            bh_history_for_ticker.append(capital_per_stock)

        return {
            'ticker': ticker,
            'final_val': final_val,
            'perf_data': perf_data,
            'individual_bh_return': individual_bh_return,
            'last_ai_action': last_ai_action,
            'buy_prob': last_buy_prob,
            'sell_prob': last_sell_prob,
            'shares_before_liquidation': shares_before_liquidation,
            'buy_hold_history': bh_history_for_ticker
        }
    finally:
        with open("logs/worker_debug.log", "a") as f:
            final_val_to_log = 'Error' if 'final_val' not in locals() else final_val
            f.write(f"Worker finished for ticker: {ticker}. Final Value: {final_val_to_log}\n")
