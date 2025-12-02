import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

# Import configuration from config.py
from config import (
    FEAT_SMA_SHORT, FEAT_SMA_LONG, FEAT_VOL_WINDOW, ATR_PERIOD
)

def fetch_training_data(ticker: str, data: pd.DataFrame, target_percentage: Optional[float] = None, class_horizon: Optional[int] = None, include_targets: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """Compute ML features from a given DataFrame."""
    print(f"  [DIAGNOSTIC] {ticker}: fetch_training_data - Initial data rows: {len(data)}")
    if data.empty or len(data) < FEAT_SMA_LONG + 10:
        print(f"  [DIAGNOSTIC] {ticker}: Skipping feature prep. Initial data has {len(data)} rows, required > {FEAT_SMA_LONG + 10}.")
        return pd.DataFrame(), []

    df = data.copy()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    
    if df.empty:
        print(f"  [DIAGNOSTIC] {ticker}: DataFrame became empty after dropping NaNs in 'Close'. Skipping feature prep.")
        return pd.DataFrame(), []
    
    # Removed duplicate check for df.empty

    # Fill missing values in other columns
    df = df.ffill().bfill()

    df["Returns"]    = df["Close"].pct_change(fill_method=None)
    df["SMA_F_S"]    = df["Close"].rolling(FEAT_SMA_SHORT).mean()
    df["SMA_F_L"]    = df["Close"].rolling(FEAT_SMA_LONG).mean()
    df["Volatility"] = df["Returns"].rolling(FEAT_VOL_WINDOW).std()

    # --- Additional Features ---
    # ATR (Average True Range)
    high = df["High"] if "High" in df.columns else None
    low  = df["Low"]  if "Low" in df.columns else None
    prev_close = df["Close"].shift(1)
    if high is not None and low is not None:
        hl = (high - low).abs()
        h_pc = (high - prev_close).abs()
        l_pc = (low  - prev_close).abs()
        tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(ATR_PERIOD).mean()
    else:
        # Fallback for ATR if High/Low are not available (though they should be after load_prices)
        ret = df["Close"].pct_change(fill_method=None)
        df["ATR"] = (ret.rolling(ATR_PERIOD).std() * df["Close"]).rolling(2).mean()
    df["ATR"] = df["ATR"].fillna(0) # Fill any NaNs from initial ATR calculation

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=14 - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=14 - 1, adjust=False).mean()
    rs = gain / loss
    df['RSI_feat'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_mid'] = df["Close"].rolling(window=20).mean()
    df['BB_std'] = df["Close"].rolling(window=20).std()
    df['BB_upper'] = df['BB_mid'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_mid'] - (df['BB_std'] * 2)

    # Stochastic Oscillator
    low_14, high_14 = df['Low'].rolling(window=14).min(), df['High'].rolling(window=14).max()
    denominator_k = (high_14 - low_14)
    df['%K'] = np.where(denominator_k != 0, (df['Close'] - low_14) / denominator_k * 100, 0)
    df['%D'] = df['%K'].rolling(window=3).mean()
    df['%K'] = df['%K'].fillna(0)
    df['%D'] = df['%D'].fillna(0)

    # Average Directional Index (ADX)
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['+DM'] = df['+DM'].fillna(0)

    # Calculate True Range (TR)
    high_low_diff = df['High'] - df['Low']
    high_prev_close_diff_abs = (df['High'] - df['Close'].shift(1)).abs()
    low_prev_close_diff_abs = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([high_low_diff, high_prev_close_diff_abs, low_prev_close_diff_abs], axis=1).max(axis=1)
    df['TR'] = df['TR'].fillna(0)

    # Calculate Smoothed DM and TR
    alpha = 1/14
    df['+DM14'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    df['TR14'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DM14'] = df['+DM14'].fillna(0)
    df['-DM14'] = df['-DM14'].fillna(0)
    df['TR14'] = df['TR14'].fillna(0)

    # Calculate Directional Index (DX)
    denominator_dx = (df['+DM14'] + df['-DM14'])
    df['DX'] = np.where(denominator_dx != 0, (abs(df['+DM14'] - df['-DM14']) / denominator_dx) * 100, 0)
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    df['ADX'] = df['ADX'].fillna(0)

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Chaikin Money Flow (CMF)
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['CMF'] = mfv.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    df['CMF'] = df['CMF'].fillna(0)

    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=12) * 100

    # Keltner Channels
    df['KC_TR'] = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['KC_ATR'] = df['KC_TR'].rolling(window=10).mean()
    df['KC_Middle'] = df["Close"].rolling(window=20).mean()
    df['KC_Upper'] = df['KC_Middle'] + (df['KC_ATR'] * 2)
    df['KC_Lower'] = df['KC_Middle'] - (df['KC_ATR'] * 2)

    # Donchian Channels
    df['DC_Upper'] = df['High'].rolling(window=20).max()
    df['DC_Lower'] = df['Low'].rolling(window=20).min()
    df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2

    # Parabolic SAR (PSAR)
    psar = df['Close'].copy()
    af = 0.02
    max_af = 0.2

    uptrend = True if df['Close'].iloc[0] > df['Open'].iloc[0] else False
    ep = df['High'].iloc[0] if uptrend else df['Low'].iloc[0]
    sar = df['Low'].iloc[0] if uptrend else df['High'].iloc[0]
    
    for i in range(1, len(df)):
        if uptrend:
            sar = sar + af * (ep - sar)
            if df['Low'].iloc[i] < sar:
                uptrend = False
                sar = ep
                ep = df['Low'].iloc[i]
                af = 0.02
            else:
                if df['High'].iloc[i] > ep:
                    ep = df['High'].iloc[i]
                    af = min(max_af, af + 0.02)
        else:
            sar = sar + af * (ep - sar)
            if df['High'].iloc[i] > sar:
                uptrend = True
                sar = ep
                ep = df['High'].iloc[i]
                af = 0.02
            else:
                if df['Low'].iloc[i] < ep:
                    ep = df['Low'].iloc[i]
                    af = min(max_af, af + 0.02)
            psar.iloc[i] = sar
    df['PSAR'] = psar

    # Accumulation/Distribution Line (ADL)
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_multiplier * df['Volume']
    df['ADL'] = mf_volume.cumsum()
    df['ADL'] = df['ADL'].fillna(0)

    # Commodity Channel Index (CCI)
    TP = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (TP - TP.rolling(window=20).mean()) / (0.015 * TP.rolling(window=20).std())
    df['CCI'] = df['CCI'].fillna(0)

    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=FEAT_VOL_WINDOW).sum() / df['Volume'].rolling(window=FEAT_VOL_WINDOW).sum()
    df['VWAP'] = df['VWAP'].fillna(df['Close'])

    # ATR Percentage
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    df['ATR_Pct'] = df['ATR_Pct'].fillna(0)

    # Chaikin Oscillator
    adl_fast = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    adl_slow = adl_fast.ewm(span=10, adjust=False).mean()
    adl_fast = adl_fast.ewm(span=3, adjust=False).mean()
    df['Chaikin_Oscillator'] = adl_fast - adl_slow
    df['Chaikin_Oscillator'] = df['Chaikin_Oscillator'].fillna(0)

    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_mf = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_mf = money_flow.where(typical_price < typical_price.shift(1), 0)
    mfi_ratio = positive_mf.rolling(window=14).sum() / negative_mf.rolling(window=14).sum()
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    df['MFI'] = df['MFI'].fillna(0)

    # OBV Moving Average
    df['OBV_SMA'] = df['OBV'].rolling(window=10).mean()
    df['OBV_SMA'] = df['OBV_SMA'].fillna(0)

    # Historical Volatility (e.g., 20-day rolling standard deviation of log returns)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Historical_Volatility'] = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252)
    df['Historical_Volatility'] = df['Historical_Volatility'].fillna(0)

    # --- Additional Financial Features (from _fetch_financial_data) ---
    financial_features = [col for col in df.columns if col.startswith('Fin_')]
    
    # Ensure these are numeric and fill NaNs if any remain
    for col in financial_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Dynamically build the list of features that are actually present in the DataFrame
    expected_technical_features = [
        "Close", "Volume", "High", "Low", "Open", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", 
        "ATR", "RSI_feat", "MACD", "MACD_signal", "BB_upper", "BB_lower", "%K", "%D", "ADX",
        "OBV", "CMF", "ROC", "KC_Upper", "KC_Lower", "DC_Upper", "DC_Lower",
        "PSAR", "ADL", "CCI", "VWAP", "ATR_Pct", "Chaikin_Oscillator", "MFI", "OBV_SMA", "Historical_Volatility",
        "Market_Momentum_SPY",
        "Sentiment_Score",
        "VIX_Index_Returns", "DXY_Index_Returns", "Gold_Futures_Returns", "Oil_Futures_Returns", "US10Y_Yield_Returns",
        "Oil_Price_Returns", "Gold_Price_Returns"
    ]
    
    present_technical_features = [col for col in expected_technical_features if col in df.columns]
    all_present_features = present_technical_features + financial_features
    
    target_cols = []
    if include_targets and target_percentage is not None and class_horizon is not None:
        df["Target"] = df["Close"].shift(-1)
        fwd = df["Close"].shift(-class_horizon)
        df["TargetClassBuy"] = ((fwd / df["Close"] - 1.0) > target_percentage).astype(float)
        df["TargetClassSell"] = ((fwd / df["Close"] - 1.0) < -target_percentage).astype(float)
        target_cols = ["Target", "TargetClassBuy", "TargetClassSell"]

    cols_for_ready = all_present_features + target_cols
    cols_for_ready_final = [col for col in cols_for_ready if col in df.columns]

    ready = df[cols_for_ready_final].dropna()
    
    final_training_features = [col for col in ready.columns if col not in target_cols]

    print(f"   ↳ {ticker}: rows after features available: {len(ready)}")
    return ready, final_training_features
