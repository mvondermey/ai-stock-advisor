import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

# Import configuration from config.py
from config import (
    USE_MODEL_GATE,
    INVESTMENT_PER_STOCK, TRANSACTION_COST, ATR_PERIOD, FEAT_SMA_SHORT,
    FEAT_SMA_LONG, FEAT_VOL_WINDOW, SEQUENCE_LENGTH,
    USE_SINGLE_REGRESSION_MODEL
)

# Import PyTorch models if available
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    from ml_models import LSTMClassifier, GRUClassifier, GRURegressor, CUDA_AVAILABLE # Import all model classes
except ImportError:
    PYTORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    # Define dummy classes
    class LSTMClassifier: pass
    class GRUClassifier: pass
    class GRURegressor: pass

class RuleTradingEnv:
    """SMA cross + ATR trailing stop/TP + risk-based sizing. Minimal ML thresholds - ranking drives portfolio decisions."""
    def __init__(self, df: pd.DataFrame, ticker: str, initial_balance: float, transaction_cost: float,
                 model=None, scaler=None, y_scaler=None, use_gate: bool = USE_MODEL_GATE,
                 feature_set: Optional[List[str]] = None,
                 horizon_days: int = 20):
        if "Close" not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column.")
        self.df = df.reset_index()
        self.ticker = ticker
        self.initial_balance = float(initial_balance)
        self.transaction_cost = float(transaction_cost)
        self.model = model  # Single regression model
        self.scaler = scaler
        self.y_scaler = y_scaler  # ✅ Store y_scaler for inverse transforming predictions
        self.use_gate = bool(use_gate) and (scaler is not None) and USE_SINGLE_REGRESSION_MODEL
        
        # DEBUG: Log environment initialization
        import sys
        sys.stderr.write(f"[ENV {ticker}] RuleTradingEnv initialized:\n")
        sys.stderr.write(f"  - use_gate (computed): {self.use_gate} (input use_gate={use_gate}, scaler={'OK' if scaler else 'None'}, regression={USE_SINGLE_REGRESSION_MODEL})\n")
        sys.stderr.write(f"  - model: {type(model).__name__ if model else 'None'}\n")
        sys.stderr.write(f"  - y_scaler: {type(y_scaler).__name__ if y_scaler else 'None'}\n")
        sys.stderr.flush()
        self.horizon_days = int(horizon_days)  # Prediction horizon for evaluation
        self.feature_set = feature_set if feature_set is not None else [
            "Close", "Volume", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", "ATR",
            "RSI_feat", "MACD", "MACD_signal", "BB_upper", "BB_lower", "%K", "%D", "ADX",
            "OBV", "CMF", "ROC", "KC_Upper", "KC_Lower", "DC_Upper", "DC_Lower",
            "PSAR", "ADL", "CCI", "VWAP", "ATR_Pct", "Chaikin_Oscillator", "MFI", "OBV_SMA", "Historical_Volatility",
            'Fin_Revenue', 'Fin_NetIncome', 'Fin_TotalAssets', 'Fin_TotalLiabilities', 'Fin_FreeCashFlow', 'Fin_EBITDA',
            'Market_Momentum_SPY'
        ]
        
        self.reset()
        self._prepare_data()

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_balance
        self.shares = 0.0
        self.max_shares_held = 0.0  # Track maximum position size
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
        
        # Track predictions for analysis
        self.all_predictions_buy = []
        self.all_predictions_sell = []
        
    def _prepare_data(self):
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

        self.df["Returns"]    = close.pct_change(fill_method=None)
        self.df["SMA_F_S"]    = close.rolling(FEAT_SMA_SHORT).mean()
        self.df["SMA_F_L"]    = close.rolling(FEAT_SMA_LONG).mean()
        self.df["Volatility"] = self.df["Returns"].rolling(FEAT_VOL_WINDOW).std()
        
        delta_feat = close.diff()
        gain_feat = (delta_feat.where(delta_feat > 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        loss_feat = (-delta_feat.where(delta_feat < 0, 0)).ewm(com=14 - 1, adjust=False).mean()
        rs_feat = gain_feat / loss_feat
        self.df['RSI_feat'] = 100 - (100 / (1 + rs_feat))

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        self.df['MACD'] = ema_12 - ema_26
        self.df['MACD_signal'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        self.df['BB_upper'] = bb_mid + (bb_std * 2)
        self.df['BB_lower'] = bb_mid - (bb_std * 2)

        low_14, high_14 = self.df['Low'].rolling(window=14).min(), self.df['High'].rolling(window=14).max()
        self.df['%K'] = (self.df['Close'] - low_14) / (high_14 - low_14) * 100
        self.df['%D'] = self.df['%K'].rolling(window=3).mean()

        self.df['up_move'] = self.df['High'] - self.df['High'].shift(1)
        self.df['down_move'] = self.df['Low'].shift(1) - self.df['Low']
        self.df['+DM'] = np.where((self.df['up_move'] > self.df['down_move']) & (self.df['up_move'] > 0), self.df['up_move'], 0)
        self.df['-DM'] = np.where((self.df['down_move'] > self.df['up_move']) & (self.df['down_move'] > 0), self.df['down_move'], 0)
        high_low_diff = self.df['High'] - self.df['Low']
        high_prev_close_diff_abs = (self.df['High'] - self.df['Close'].shift(1)).abs()
        low_prev_close_diff_abs = (self.df['Low'] - self.df['Close'].shift(1)).abs()
        self.df['TR'] = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
        alpha = 1/14
        self.df['+DM14'] = self.df['+DM'].ewm(alpha=alpha, adjust=False).mean()
        self.df['-DM14'] = self.df['-DM'].ewm(alpha=alpha, adjust=False).mean()
        self.df['TR14'] = self.df['TR'].ewm(alpha=alpha, adjust=False).mean()
        self.df['DX'] = (abs(self.df['+DM14'] - self.df['-DM14']) / (self.df['+DM14'] + self.df['-DM14'])) * 100
        self.df['DX'] = self.df['DX'].fillna(0)
        self.df['ADX'] = self.df['DX'].ewm(alpha=alpha, adjust=False).mean()
        self.df['ADX'] = self.df['ADX'].fillna(0)
        self.df['+DM'] = self.df['+DM'].fillna(0)
        self.df['-DM'] = self.df['-DM'].fillna(0)
        self.df['TR'] = self.df['TR'].fillna(0)
        self.df['+DM14'] = self.df['+DM14'].fillna(0)
        self.df['-DM14'] = self.df['-DM14'].fillna(0)
        self.df['TR14'] = self.df['TR14'].fillna(0)
        self.df['%K'] = self.df['%K'].fillna(0)
        self.df['%D'] = self.df['%D'].fillna(0)

        self.df['OBV'] = (np.sign(self.df['Close'].diff()) * self.df['Volume']).fillna(0).cumsum()

        mfv = ((self.df['Close'] - self.df['Low']) - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low']) * self.df['Volume']
        self.df['CMF'] = mfv.rolling(window=20).sum() / self.df['Volume'].rolling(window=20).sum()
        self.df['CMF'] = self.df['CMF'].fillna(0)

        self.df['ROC'] = self.df['Close'].pct_change(periods=12) * 100

        self.df['KC_TR'] = pd.concat([self.df['High'] - self.df['Low'], (self.df['High'] - self.df['Close'].shift(1)).abs(), (self.df['Low'] - self.df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        self.df['KC_ATR'] = self.df['KC_TR'].rolling(window=10).mean()
        self.df['KC_Middle'] = self.df['Close'].rolling(window=20).mean()
        self.df['KC_Upper'] = self.df['KC_Middle'] + (self.df['KC_ATR'] * 2)
        self.df['KC_Lower'] = self.df['KC_Middle'] - (self.df['KC_ATR'] * 2)

        self.df['DC_Upper'] = self.df['High'].rolling(window=20).max()
        self.df['DC_Lower'] = self.df['Low'].rolling(window=20).min()
        self.df['DC_Middle'] = (self.df['DC_Upper'] + self.df['DC_Lower']) / 2

        psar = self.df['Close'].copy()
        af = 0.02
        max_af = 0.2

        uptrend = True if self.df['Close'].iloc[0] > self.df['Open'].iloc[0] else False
        ep = self.df['High'].iloc[0] if uptrend else self.df['Low'].iloc[0]
        sar = self.df['Low'].iloc[0] if uptrend else self.df['High'].iloc[0]
        
        for i in range(1, len(self.df)):
            if uptrend:
                sar = sar + af * (ep - sar)
                if self.df['Low'].iloc[i] < sar:
                    uptrend = False
                    sar = ep
                    ep = self.df['Low'].iloc[i]
                    af = 0.02
                else:
                    if self.df['High'].iloc[i] > ep:
                        ep = self.df['High'].iloc[i]
                        af = min(max_af, af + 0.02)
            else:
                sar = sar + af * (ep - sar)
                if self.df['High'].iloc[i] > sar:
                    uptrend = True
                    sar = ep
                    ep = self.df['High'].iloc[i]
                    af = 0.02
                else:
                    if self.df['Low'].iloc[i] < ep:
                        ep = self.df['Low'].iloc[i]
                        af = min(max_af, af + 0.02)
            psar.iloc[i] = sar
        self.df['PSAR'] = psar

        mf_multiplier = ((self.df['Close'] - self.df['Low']) - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low'])
        mf_volume = mf_multiplier * self.df['Volume']
        self.df['ADL'] = mf_volume.cumsum()
        self.df['ADL'] = self.df['ADL'].fillna(0)

        TP = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['CCI'] = (TP - TP.rolling(window=20).mean()) / (0.015 * TP.rolling(window=20).std())
        self.df['CCI'] = self.df['CCI'].fillna(0)

        self.df['VWAP'] = (self.df['Close'] * self.df['Volume']).rolling(window=FEAT_VOL_WINDOW).sum() / self.df['Volume'].rolling(window=FEAT_VOL_WINDOW).sum()
        self.df['VWAP'] = self.df['VWAP'].fillna(self.df['Close'])

        if "ATR" not in self.df.columns:
            high = self.df["High"] if "High" in self.df.columns else None
            low  = self.df["Low"]  if "Low" in self.df.columns else None
            prev_close = self.df["Close"].shift(1)
            if high is not None and low is not None:
                hl = (high - low).abs()
                h_pc = (high - prev_close).abs()
                l_pc = (low  - prev_close).abs()
                tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
                self.df["ATR"] = tr.rolling(ATR_PERIOD).mean()
            else:
                ret = self.df["Close"].pct_change(fill_method=None)
                self.df["ATR"] = (ret.rolling(ATR_PERIOD).std() * close).rolling(2).mean()
            self.df["ATR"] = self.df["ATR"].fillna(0)

        self.df['ATR_Pct'] = (self.df['ATR'] / self.df['Close']) * 100
        self.df['ATR_Pct'] = self.df['ATR_Pct'].fillna(0)

        mf_multiplier_co = ((self.df['Close'] - self.df['Low']) - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low'])
        adl_fast = (mf_multiplier_co * self.df['Volume']).ewm(span=3, adjust=False).mean()
        adl_slow = (mf_multiplier_co * self.df['Volume']).ewm(span=10, adjust=False).mean()
        self.df['Chaikin_Oscillator'] = adl_fast - adl_slow
        self.df['Chaikin_Oscillator'] = self.df['Chaikin_Oscillator'].fillna(0)

        typical_price_mfi = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow_mfi = typical_price_mfi * self.df['Volume']
        positive_mf_mfi = money_flow_mfi.where(typical_price_mfi > typical_price_mfi.shift(1), 0)
        negative_mf_mfi = money_flow_mfi.where(typical_price_mfi < typical_price_mfi.shift(1), 0)
        mfi_ratio = positive_mf_mfi.rolling(window=14).sum() / negative_mf_mfi.rolling(window=14).sum()
        self.df['MFI'] = 100 - (100 / (1 + mfi_ratio))
        self.df['MFI'] = self.df['MFI'].fillna(0)

        self.df['OBV_SMA'] = self.df['OBV'].rolling(window=10).mean()
        self.df['OBV_SMA'] = self.df['OBV_SMA'].fillna(0)

        self.df['Log_Returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['Historical_Volatility'] = self.df['Log_Returns'].rolling(window=20).std() * np.sqrt(252)
        self.df['Historical_Volatility'] = self.df['Historical_Volatility'].fillna(0)

    def _date_at(self, i: int) -> str:
        if "Date" in self.df.columns:
            return str(self.df.loc[i, "Date"])
        return str(i)

    def _get_model_prediction(self, i: int) -> float:
        """
        Get prediction from single regression model at step i.
        Returns predicted return percentage (e.g., 0.05 = 5%).
        """
        # DEBUG: Log first prediction attempt
        if i == 0:
            print(f"  [DEBUG {self.ticker}] Step 0: use_gate={self.use_gate}, model={type(self.model).__name__ if self.model else 'None'}, scaler={type(self.scaler).__name__ if self.scaler else 'None'}, y_scaler={type(self.y_scaler).__name__ if self.y_scaler else 'None'}")

        if not self.use_gate or self.model is None:
            if i == 0:
                print(f"  [DEBUG {self.ticker}] Step 0: Returning 0.0 early! use_gate={self.use_gate}, model_is_None={self.model is None}")
            return 0.0
        
        model_feature_names = self.scaler.feature_names_in_ if hasattr(self.scaler, 'feature_names_in_') else self.feature_set
        
        # Create a dictionary with default 0.0 values for all required features
        feature_dict = {feature: 0.0 for feature in model_feature_names}
        
        # Update with available values from the DataFrame
        row = self.df.iloc[i]
        for feature in model_feature_names:
            if feature in row.index:
                try:
                    feature_dict[feature] = row[feature]
                except (KeyError, IndexError):
                    # Feature exists in index but access failed, use default
                    pass
            # If feature not in row.index, it stays as 0.0 from initialization
        
        # Create DataFrame with one row
        X_df = pd.DataFrame([list(feature_dict.values())], columns=model_feature_names)
        
        # Ensure all values are numeric
        X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        if X_df.isnull().all().any():
            print(f"  [{self.ticker}] Critical: Feature column is all NaN after fillna at step {i}. Skipping prediction.")
            return 0.0

        try:
            if PYTORCH_AVAILABLE and isinstance(model, (LSTMClassifier, GRUClassifier, GRURegressor)):
                start_idx = max(0, i - SEQUENCE_LENGTH + 1)
                end_idx = i + 1
                
                # Filter model_feature_names to only include features that exist in the DataFrame
                available_features = [f for f in model_feature_names if f in self.df.columns]
                
                if not available_features:
                    print(f"  [{self.ticker}] Warning: No model features available in DataFrame at step {i}. Skipping prediction.")
                    return 0.0
                
                # Create a DataFrame with all required features, filling missing ones with 0.0
                historical_data_for_seq = self.df.loc[start_idx:end_idx-1].copy()
                
                # Ensure all required features exist in the historical data
                for feature in model_feature_names:
                    if feature not in historical_data_for_seq.columns:
                        historical_data_for_seq[feature] = 0.0
                
                # Select only the features we need, in the correct order
                historical_data_for_seq = historical_data_for_seq[model_feature_names].copy()
                
                # Convert to numeric and fill NaNs
                for col in historical_data_for_seq.columns:
                    historical_data_for_seq[col] = pd.to_numeric(historical_data_for_seq[col], errors='coerce').fillna(0.0)

                X_scaled_seq = self.scaler.transform(historical_data_for_seq)
                
                # Pad sequence if its length is less than SEQUENCE_LENGTH
                if X_scaled_seq.shape[0] < SEQUENCE_LENGTH:
                    padding_needed = SEQUENCE_LENGTH - X_scaled_seq.shape[0]
                    padding_zeros = np.zeros((padding_needed, X_scaled_seq.shape[1]))
                    X_scaled_seq = np.vstack((padding_zeros, X_scaled_seq))

                X_tensor = torch.tensor(X_scaled_seq, dtype=torch.float32).unsqueeze(0)
                
                device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
                X_tensor = X_tensor.to(device)

                model.to(device)

                model.eval()
                with torch.no_grad():
                    output = model(X_tensor)
                    scaled_prediction = float(output.cpu().numpy()[0][0])
                    
                    # ✅ Inverse transform if y_scaler is available (for regression)
                    if self.y_scaler is not None:
                        prediction = self.y_scaler.inverse_transform([[scaled_prediction]])[0][0]
                        # DEBUG: Log first few predictions
                        if i < 3:
                            print(f"  [DEBUG {self.ticker}] Step {i}: scaled_pred={scaled_prediction:.6f}, inverse_pred={prediction:.6f}, scaler_min={self.y_scaler.data_min_[0]:.4f}, scaler_max={self.y_scaler.data_max_[0]:.4f}")
                        return prediction
                    else:
                        return scaled_prediction
            else:
                # For sklearn models
                X_scaled_np = self.scaler.transform(historical_data_for_seq)
                X = pd.DataFrame(X_scaled_np, columns=model_feature_names) if model_feature_names else pd.DataFrame(X_scaled_np)

                # Single regression model - predict return percentage
                scaled_prediction = float(self.model.predict(X)[0])

                # ✅ Inverse transform with y_scaler for regression
                if self.y_scaler is not None:
                    prediction = self.y_scaler.inverse_transform([[scaled_prediction]])[0][0]
                    return prediction
                else:
                    return scaled_prediction
        except Exception as e:
            print(f"  [{self.ticker}] Error in model prediction at step {i}: {e}")
            return 0.0


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
        self.max_shares_held = max(self.max_shares_held, self.shares)  # Track max position
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
        """Simplified: Buy at start, hold until end of period."""
        if self.current_step < 1:
            # Buy at the beginning of the period
            if len(self.df) > 0:
                first_row = self.df.iloc[0]
                first_price = float(first_row["Close"])
                first_date = self._date_at(0)
                atr = float(first_row.get("ATR", np.nan)) if pd.notna(first_row.get("ATR", np.nan)) else None

                # Buy immediately (no AI decision needed for selection-based system)
                self._buy(first_price, atr, first_date)
                self.last_ai_action = "BUY"
                self.last_buy_prob = 1.0  # Full confidence in selection
                self.last_sell_prob = 0.0

            self.portfolio_history.append(self.initial_balance)
            self.current_step = len(self.df)  # Skip to end
            return True

        # Hold until end of period
        if self.current_step >= len(self.df):
            return True

        # Track portfolio value but don't trade
        row = self.df.iloc[self.current_step]
        price = float(row["Close"])
        port_val = self.cash + self.shares * price
        self.portfolio_history.append(port_val)
        self.current_step += 1

        self.last_ai_action = "HOLD"
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
        
        # Return max_shares_held instead of shares_before_liquidation for better visibility
        return self.portfolio_history[-1], self.trade_log, self.last_ai_action, self.last_buy_prob, self.last_sell_prob, self.max_shares_held
