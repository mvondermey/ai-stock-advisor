# data_utils.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path # Import Path for _ensure_dir
from utils import _to_utc
import yfinance as yf
import time

# Import config values
try:
    from config import (
        PAUSE_BETWEEN_YF_CALLS, DATA_PROVIDER, USE_YAHOO_FALLBACK, DATA_CACHE_DIR, CACHE_DAYS, 
        TWELVEDATA_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_AVAILABLE, TWELVEDATA_SDK_AVAILABLE,
        FEAT_SMA_SHORT, FEAT_SMA_LONG, FEAT_VOL_WINDOW, ATR_PERIOD
    )
except ImportError:
    # Fallback values if config is not available
    PAUSE_BETWEEN_YF_CALLS = 0.5
    DATA_PROVIDER = 'yahoo'
    USE_YAHOO_FALLBACK = True
    DATA_CACHE_DIR = Path("data_cache")
    CACHE_DAYS = 7
    TWELVEDATA_API_KEY = None
    ALPACA_API_KEY = None
    ALPACA_SECRET_KEY = None
    ALPACA_AVAILABLE = False
    TWELVEDATA_SDK_AVAILABLE = False
    FEAT_SMA_SHORT = 5
    FEAT_SMA_LONG = 20
    FEAT_VOL_WINDOW = 10
    ATR_PERIOD = 14

# Optional Stooq provider
try:
    from pandas_datareader import data as pdr
except Exception:
    pdr = None

# Define technical indicators calculation function here to avoid circular imports
def _calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates technical indicators and adds them to the DataFrame."""
    close = df["Close"]
    high = df["High"] if "High" in df.columns else None
    low  = df["Low"]  if "Low" in df.columns else None
    prev_close = close.shift(1)

    # ATR for risk management
    if high is not None and low is not None:
        hl = (high - low).abs()
        h_pc = (high - prev_close).abs()
        l_pc = (low  - prev_close).abs()
        tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(ATR_PERIOD).mean()
    else:
        ret = close.pct_change(fill_method=None)
        df["ATR"] = (ret.rolling(ATR_PERIOD).std() * close).rolling(2).mean()
    
    # Low-volatility filter reference: rolling median ATR
    df['ATR_MED'] = df['ATR'].rolling(50).median()

    # --- Features for ML Gate ---
    df["Returns"]    = close.pct_change(fill_method=None)
    df["SMA_F_S"]    = close.rolling(FEAT_SMA_SHORT).mean()
    df["SMA_F_L"]    = close.rolling(FEAT_SMA_LONG).mean()
    df["Volatility"] = df["Returns"].rolling(FEAT_VOL_WINDOW).std()
    
    # RSI for features
    delta_feat = close.diff()
    gain_feat = (delta_feat.where(delta_feat > 0, 0)).ewm(com=14 - 1, adjust=False).mean()
    loss_feat = (-delta_feat.where(delta_feat < 0, 0)).ewm(com=14 - 1, adjust=False).mean()
    rs_feat = gain_feat / loss_feat
    df['RSI_feat'] = 100 - (100 / (1 + rs_feat))

    # MACD for features
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands for features
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['BB_upper'] = bb_mid + (bb_std * 2)
    df['BB_lower'] = bb_mid - (bb_std * 2)

    # Stochastic Oscillator
    low_14, high_14 = df['Low'].rolling(window=14).min(), df['High'].rolling(window=14).max()
    df['%K'] = (df['Close'] - low_14) / (high_14 - low_14) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Average Directional Index (ADX)
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    hl_diff = (df['High'] - df['Low']).abs()
    h_pc_diff = (df['High'] - df['Close'].shift(1)).abs()
    l_pc_diff = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = pd.concat([hl_diff, h_pc_diff, l_pc_diff], axis=1).max(axis=1)
    alpha = 1/14
    df['+DM14'] = df['+DM'].ewm(alpha=alpha, adjust=False).mean()
    df['-DM14'] = df['-DM'].ewm(alpha=alpha, adjust=False).mean()
    df['TR14'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['DX'] = (abs(df['+DM14'] - df['-DM14']) / (df['+DM14'] + df['-DM14'])) * 100
    df['DX'] = df['DX'].fillna(0)
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    df['ADX'] = df['ADX'].fillna(0)
    df['+DM'] = df['+DM'].fillna(0)
    df['-DM'] = df['-DM'].fillna(0)
    df['TR'] = df['TR'].fillna(0)
    df['+DM14'] = df['+DM14'].fillna(0)
    df['-DM14'] = df['-DM14'].fillna(0)
    df['TR14'] = df['TR14'].fillna(0)
    df['%K'] = df['%K'].fillna(0)
    df['%D'] = df['%D'].fillna(0)

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Chaikin Money Flow (CMF)
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    df['CMF'] = mfv.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    df['CMF'] = df['CMF'].fillna(0)

    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=12) * 100
    df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
    df['ROC_60'] = df['Close'].pct_change(periods=60) * 100

    # Chande Momentum Oscillator (CMO)
    cmo_period = 14
    df['cmo_diff'] = df['Close'].diff()
    df['cmo_up'] = df['cmo_diff'].apply(lambda x: x if x > 0 else 0)
    df['cmo_down'] = df['cmo_diff'].apply(lambda x: abs(x) if x < 0 else 0)
    df['cmo_sum_up'] = df['cmo_up'].rolling(window=cmo_period).sum()
    df['cmo_sum_down'] = df['cmo_down'].rolling(window=cmo_period).sum()
    df['CMO'] = ((df['cmo_sum_up'] - df['cmo_sum_down']) / (df['cmo_sum_up'] + df['cmo_sum_down'])) * 100
    df['CMO'] = df['CMO'].fillna(0)

    # Kaufman's Adaptive Moving Average (KAMA)
    kama_period = 10
    fast_ema_const = 2 / (2 + 1)
    slow_ema_const = 2 / (30 + 1)
    df['kama_change'] = abs(df['Close'] - df['Close'].shift(kama_period))
    df['kama_volatility'] = df['Close'].diff().abs().rolling(window=kama_period).sum()
    df['kama_er'] = df['kama_change'] / df['kama_volatility']
    df['kama_er'] = df['kama_er'].fillna(0)
    df['kama_sc'] = (df['kama_er'] * (fast_ema_const - slow_ema_const) + slow_ema_const)**2
    df['KAMA'] = np.nan
    df.iloc[kama_period-1, df.columns.get_loc('KAMA')] = df['Close'].iloc[kama_period-1]
    for i in range(kama_period, len(df)):
        df.iloc[i, df.columns.get_loc('KAMA')] = df.iloc[i-1, df.columns.get_loc('KAMA')] + df.iloc[i, df.columns.get_loc('kama_sc')] * (df.iloc[i, df.columns.get_loc('Close')] - df.iloc[i-1, df.columns.get_loc('KAMA')])
    df['KAMA'] = df['KAMA'].ffill().bfill().fillna(df['Close'])

    # Elder's Force Index (EFI)
    efi_period = 13
    df['EFI'] = (df['Close'].diff() * df['Volume']).ewm(span=efi_period, adjust=False).mean()
    df['EFI'] = df['EFI'].fillna(0)

    # Keltner Channels
    df['KC_TR'] = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['KC_ATR'] = df['KC_TR'].rolling(window=10).mean()
    df['KC_Middle'] = df['Close'].rolling(window=20).mean()
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
            if df.iloc[i, df.columns.get_loc('Low')] < sar:
                uptrend = False
                sar = ep
                ep = df.iloc[i, df.columns.get_loc('Low')]
                af = 0.02
            else:
                if df.iloc[i, df.columns.get_loc('High')] > ep:
                    ep = df.iloc[i, df.columns.get_loc('High')]
                    af = min(max_af, af + 0.02)
        else:
            sar = sar + af * (ep - sar)
            if df.iloc[i, df.columns.get_loc('High')] > sar:
                uptrend = True
                sar = ep
                ep = df.iloc[i, df.columns.get_loc('High')]
                af = 0.02
            else:
                if df.iloc[i, df.columns.get_loc('Low')] < ep:
                    ep = df.iloc[i, df.columns.get_loc('Low')]
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
    if "ATR" not in df.columns:
        if high is not None and low is not None:
            hl = (high - low).abs()
            h_pc = (high - prev_close).abs()
            l_pc = (low  - prev_close).abs()
            tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
            df["ATR"] = tr.rolling(ATR_PERIOD).mean()
        else:
            ret = df["Close"].pct_change(fill_method=None)
            df["ATR"] = (ret.rolling(ATR_PERIOD).std() * df["Close"]).rolling(2).mean()
        df["ATR"] = df["ATR"].fillna(0)

    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100
    df['ATR_Pct'] = df['ATR_Pct'].fillna(0)

    # Chaikin Oscillator
    mf_multiplier_co = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    adl_fast = (mf_multiplier_co * df['Volume']).ewm(span=3, adjust=False).mean()
    adl_slow = (mf_multiplier_co * df['Volume']).ewm(span=10, adjust=False).mean()
    df['Chaikin_Oscillator'] = adl_fast - adl_slow
    df['Chaikin_Oscillator'] = df['Chaikin_Oscillator'].fillna(0)

    # Money Flow Index (MFI)
    typical_price_mfi = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow_mfi = typical_price_mfi * df['Volume']
    positive_mf_mfi = money_flow_mfi.where(typical_price_mfi > typical_price_mfi.shift(1), 0)
    negative_mf_mfi = money_flow_mfi.where(typical_price_mfi < typical_price_mfi.shift(1), 0)
    mfi_ratio = positive_mf_mfi.rolling(window=14).sum() / negative_mf_mfi.rolling(window=14).sum()
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))
    df['MFI'] = df['MFI'].fillna(0)

    # OBV Moving Average
    df['OBV_SMA'] = df['OBV'].rolling(window=10).mean()
    df['OBV_SMA'] = df['OBV_SMA'].fillna(0)

    # Historical Volatility (e.g., 20-day rolling standard deviation of log returns)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Historical_Volatility'] = df['Log_Returns'].rolling(window=20).std() * np.sqrt(252)
    df['Historical_Volatility'] = df['Historical_Volatility'].fillna(0)
    
    # ========================================
    # NEW: Momentum & Trend Features for AI
    # ========================================
    
    # 1. Multi-timeframe momentum (returns over different periods)
    df['Momentum_3d'] = (df['Close'] / df['Close'].shift(3) - 1.0).fillna(0)
    df['Momentum_5d'] = (df['Close'] / df['Close'].shift(5) - 1.0).fillna(0)
    df['Momentum_10d'] = (df['Close'] / df['Close'].shift(10) - 1.0).fillna(0)
    df['Momentum_20d'] = (df['Close'] / df['Close'].shift(20) - 1.0).fillna(0)
    df['Momentum_40d'] = (df['Close'] / df['Close'].shift(40) - 1.0).fillna(0)  # Reduced from 60 to 40
    
    # 2. Trend strength (distance from moving averages)
    df['Dist_From_SMA10'] = (df['Close'] / df['Close'].rolling(10, min_periods=5).mean() - 1.0).fillna(0)
    df['Dist_From_SMA20'] = (df['Close'] / df['Close'].rolling(20, min_periods=10).mean() - 1.0).fillna(0)
    df['Dist_From_SMA50'] = (df['Close'] / df['Close'].rolling(50, min_periods=20).mean() - 1.0).fillna(0)
    # Removed SMA200 to reduce data loss - 200 days is too long for short-term trading
    
    # 3. Moving average slopes (trend direction)
    sma20 = df['Close'].rolling(20, min_periods=10).mean()
    sma50 = df['Close'].rolling(50, min_periods=20).mean()
    df['SMA20_Slope'] = (sma20 / sma20.shift(5) - 1.0).fillna(0)
    df['SMA50_Slope'] = (sma50 / sma50.shift(10) - 1.0).fillna(0)
    
    # 4. Price acceleration (momentum of momentum)
    df['Price_Accel_5d'] = (df['Momentum_5d'] - df['Momentum_5d'].shift(1)).fillna(0)
    df['Price_Accel_20d'] = (df['Momentum_20d'] - df['Momentum_20d'].shift(5)).fillna(0)
    
    # 5. Volatility regime (current vs historical)
    df['Vol_Regime'] = (df['Volatility'] / df['Volatility'].rolling(30, min_periods=10).mean()).fillna(1.0)
    df['Vol_Spike'] = (df['Volatility'] / df['Volatility'].shift(1) - 1.0).fillna(0)
    
    # 6. Volume momentum
    df['Volume_Ratio_5d'] = (df['Volume'] / df['Volume'].rolling(5, min_periods=1).mean()).fillna(1.0)
    df['Volume_Ratio_20d'] = (df['Volume'] / df['Volume'].rolling(20, min_periods=1).mean()).fillna(1.0)
    df['Volume_Trend'] = (df['Volume'].rolling(5, min_periods=1).mean() / df['Volume'].rolling(20, min_periods=1).mean()).fillna(1.0)
    
    # 7. High-Low range expansion/contraction
    df['Range_Expansion'] = ((df['High'] - df['Low']) / df['Close']).fillna(0)
    df['Range_vs_Avg'] = (df['Range_Expansion'] / df['Range_Expansion'].rolling(20, min_periods=5).mean()).fillna(1.0)
    
    # 8. Consecutive up/down days (streak detection)
    df['Daily_Direction'] = np.where(df['Close'] > df['Close'].shift(1), 1, 
                                     np.where(df['Close'] < df['Close'].shift(1), -1, 0))
    df['Streak'] = 0
    streak = 0
    for i in range(len(df)):
        if df.iloc[i, df.columns.get_loc('Daily_Direction')] == 0:
            streak = 0
        elif i == 0:
            streak = df.iloc[i, df.columns.get_loc('Daily_Direction')]
        elif df.iloc[i, df.columns.get_loc('Daily_Direction')] == df.iloc[i-1, df.columns.get_loc('Daily_Direction')]:
            streak += df.iloc[i, df.columns.get_loc('Daily_Direction')]
        else:
            streak = df.iloc[i, df.columns.get_loc('Daily_Direction')]
        df.iloc[i, df.columns.get_loc('Streak')] = streak
    
    # 9. Momentum divergence (price vs RSI)
    if 'RSI_feat' in df.columns:
        price_change = df['Close'] / df['Close'].shift(10) - 1.0
        rsi_change = df['RSI_feat'] - df['RSI_feat'].shift(10)
        df['Momentum_Divergence'] = (price_change * rsi_change).fillna(0)
    else:
        df['Momentum_Divergence'] = 0.0
    
    # 10. Trend consistency (% of days above SMA in last 20 days)
    df['Days_Above_SMA20'] = (df['Close'] > df['Close'].rolling(20, min_periods=10).mean()).rolling(20, min_periods=10).sum() / 20.0
    df['Days_Above_SMA20'] = df['Days_Above_SMA20'].fillna(0.5)
    
    return df

# Define financial data fetching functions here to avoid circular imports
def _fetch_financial_data(ticker: str) -> pd.DataFrame:
    """Fetch key financial metrics from yfinance and prepare them for merging."""
    time.sleep(PAUSE_BETWEEN_YF_CALLS)
    yf_ticker = yf.Ticker(ticker)
    
    financial_data = {}
    
    try:
        income_statement = yf_ticker.quarterly_income_stmt
        if not income_statement.empty:
            metrics = ['Total Revenue', 'Net Income', 'EBITDA']
            for metric in metrics:
                if metric in income_statement.index:
                    financial_data[metric] = income_statement.loc[metric]
    except Exception as e:
        print(f"  ⚠️ Could not fetch income statement for {ticker}: {e}")

    try:
        balance_sheet = yf_ticker.quarterly_balance_sheet
        if not balance_sheet.empty:
            metrics = ['Total Assets', 'Total Liabilities']
            for metric in metrics:
                if metric in balance_sheet.index:
                    financial_data[metric] = balance_sheet.loc[metric]
    except Exception as e:
        print(f"  ⚠️ Could not fetch balance sheet for {ticker}: {e}")

    try:
        cash_flow = yf_ticker.quarterly_cash_flow
        if not cash_flow.empty:
            metrics = ['Free Cash Flow']
            for metric in metrics:
                if metric in cash_flow.index:
                    financial_data[metric] = cash_flow.loc[metric]
    except Exception as e:
        print(f"  ⚠️ Could not fetch cash flow for {ticker}: {e}")

    if not financial_data:
        return pd.DataFrame()

    df_financial = pd.DataFrame(financial_data).T
    df_financial.index.name = 'Metric'
    df_financial = df_financial.T
    df_financial.index = pd.to_datetime(df_financial.index, utc=True)
    df_financial.index.name = "Date"
    
    df_financial = df_financial.rename(columns={
        'Total Revenue': 'Fin_Revenue',
        'Net Income': 'Fin_NetIncome',
        'Total Assets': 'Fin_TotalAssets',
        'Total Liabilities': 'Fin_TotalLiabilities',
        'Free Cash Flow': 'Fin_FreeCashFlow',
        'EBITDA': 'Fin_EBITDA'
    })
    
    for col in df_financial.columns:
        df_financial[col] = pd.to_numeric(df_financial[col], errors='coerce')

    return df_financial.sort_index()

def _fetch_financial_data_from_alpaca(ticker: str) -> pd.DataFrame:
    """Placeholder for fetching financial metrics from Alpaca."""
    return pd.DataFrame()

# Assuming these are defined in config.py or passed as arguments
# For now, hardcode or import from a config if available
# from config import DATA_CACHE_DIR, DATA_PROVIDER, USE_YAHOO_FALLBACK, CACHE_DAYS, TWELVEDATA_API_KEY, ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_AVAILABLE, TWELVEDATA_SDK_AVAILABLE, FEAT_SMA_LONG, FEAT_SMA_SHORT, FEAT_VOL_WINDOW, ATR_PERIOD, INVESTMENT_PER_STOCK, TRANSACTION_COST, USE_MODEL_GATE, MIN_PROBA_BUY, MIN_PROBA_SELL, TARGET_PERCENTAGE, CLASS_HORIZON, SEQUENCE_LENGTH, PYTORCH_AVAILABLE, CUDA_AVAILABLE, SHAP_AVAILABLE, SAVE_PLOTS, SEED

# Placeholder for config values if not imported
DATA_CACHE_DIR = Path("data_cache")
DATA_PROVIDER = 'yahoo'
USE_YAHOO_FALLBACK = True
CACHE_DAYS = 7
TWELVEDATA_API_KEY = "YOUR_DEFAULT_KEY_OR_EMPTY_STRING"
ALPACA_API_KEY = None
ALPACA_SECRET_KEY = None
ALPACA_AVAILABLE = False
TWELVEDATA_SDK_AVAILABLE = False
FEAT_SMA_LONG = 20
FEAT_SMA_SHORT = 5
FEAT_VOL_WINDOW = 10
ATR_PERIOD = 14
INVESTMENT_PER_STOCK = 15000.0
TRANSACTION_COST = 0.001
USE_MODEL_GATE = True
MIN_PROBA_BUY = 0.20
MIN_PROBA_SELL = 0.20
TARGET_PERCENTAGE = 0.008
CLASS_HORIZON = 5
SEQUENCE_LENGTH = 32
PYTORCH_AVAILABLE = False
CUDA_AVAILABLE = False
SHAP_AVAILABLE = False
SAVE_PLOTS = False
SEED = 42


# _ensure_dir moved to utils.py to avoid duplication
from utils import _ensure_dir

CLASS_HORIZON           = 5          # days ahead for classification target
def load_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download and clean data from the selected provider, with an improved local caching mechanism."""
    _ensure_dir(DATA_CACHE_DIR)
    cache_file = DATA_CACHE_DIR / f"{ticker}.csv"
    financial_cache_file = DATA_CACHE_DIR / f"{ticker}_financials.csv"
    
    # --- Check price cache first ---
    price_df = pd.DataFrame()
    if cache_file.exists():
        file_mod_time = datetime.fromtimestamp(cache_file.stat().st_mtime, timezone.utc)
        if (datetime.now(timezone.utc) - file_mod_time) < timedelta(days=1):
            try:
                cached_df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                if cached_df.index.tzinfo is None:
                    cached_df.index = cached_df.index.tz_localize('UTC')
                else:
                    cached_df.index = cached_df.index.tz_convert('UTC')
                price_df = cached_df.loc[(cached_df.index >= _to_utc(start)) & (cached_df.index <= _to_utc(end))].copy()
            except Exception as e:
                print(f"⚠️ Could not read or slice price cache file for {ticker}: {e}. Refetching prices.")

    # --- If not in price cache or cache is old, fetch a broad range of data ---
    if price_df.empty:
        fetch_start = datetime.now(timezone.utc) - timedelta(days=1000) # Fetch a generous amount of data
        fetch_end = datetime.now(timezone.utc)
        start_utc = _to_utc(fetch_start)
        end_utc   = _to_utc(fetch_end)
        
        provider = DATA_PROVIDER.lower()
        
        if provider == 'twelvedata':
            if not TWELVEDATA_API_KEY:
                print("⚠️ TwelveData is selected but API key is missing. Falling back to Yahoo.")
                provider = 'yahoo'
            else:
                twelvedata_df = _fetch_from_twelvedata(ticker, start_utc, end_utc)
                if not twelvedata_df.empty:
                    price_df = twelvedata_df.copy()
                elif USE_YAHOO_FALLBACK:
                    print(f"  ℹ️ TwelveData fetch failed for {ticker}. Trying Yahoo Finance fallback...")
                    try:
                        downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                        if downloaded_df is not None and not downloaded_df.empty:
                            price_df = downloaded_df.dropna()
                        else:
                            print(f"  ⚠️ Yahoo Finance fallback returned empty data for {ticker}.")
                    except Exception as e:
                        print(f"  ❌ Yahoo Finance fallback failed for {ticker}: {e}")

        elif provider == 'alpaca':
            if not ALPACA_AVAILABLE or not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                print("⚠️ Alpaca is selected but API keys are missing or SDK not available. Falling back to Yahoo.")
                provider = 'yahoo'
            else:
                alpaca_df = _fetch_from_alpaca(ticker, start_utc, end_utc)
                if not alpaca_df.empty:
                    price_df = alpaca_df.copy()
                elif USE_YAHOO_FALLBACK:
                    print(f"  ℹ️ Alpaca fetch failed for {ticker}. Trying Yahoo Finance fallback...")
                    try:
                        downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                        if downloaded_df is not None and not downloaded_df.empty:
                            price_df = downloaded_df.dropna()
                        else:
                            print(f"  ⚠️ Yahoo Finance fallback returned empty data for {ticker}.")
                    except Exception as e:
                        print(f"  ❌ Yahoo Finance fallback failed for {ticker}: {e}")
        
        elif provider == 'stooq':
            stooq_df = _fetch_from_stooq(ticker, start_utc, end_utc)
            if stooq_df.empty and not ticker.upper().endswith('.US'):
                stooq_df = _fetch_from_stooq(f"{ticker}.US", start_utc, end_utc)
            if not stooq_df.empty:
                price_df = stooq_df.copy()
            elif USE_YAHOO_FALLBACK:
                print(f"  ℹ️ Stooq data for {ticker} empty. Falling back to Yahoo.")
                try:
                    downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                    if downloaded_df is not None and not downloaded_df.empty:
                        price_df = downloaded_df.dropna()
                    else:
                        print(f"  ⚠️ Yahoo Finance fallback returned empty data for {ticker}.")
                except Exception as e:
                    print(f"  ❌ Yahoo Finance fallback (after Stooq) failed for {ticker}: {e}")
        
        if price_df.empty: # If previous provider failed or was yahoo
            try:
                downloaded_df = yf.download(ticker, start=start_utc, end=end_utc, auto_adjust=True, progress=False)
                if downloaded_df is not None and not downloaded_df.empty:
                    price_df = downloaded_df.dropna()
                else:
                    print(f"  ⚠️ Final Yahoo download attempt returned empty data for {ticker}.")
            except Exception as e:
                print(f"  ❌ Final Yahoo download attempt failed for {ticker}: {e}")
            if price_df.empty and pdr is not None and DATA_PROVIDER.lower() != 'stooq':
                print(f"  ℹ️ Yahoo data for {ticker} empty. Falling back to Stooq.")
                stooq_df = _fetch_from_stooq(ticker, start_utc, end_utc)
                if stooq_df.empty and not ticker.upper().endswith('.US'):
                    stooq_df = _fetch_from_stooq(f"{ticker}.US", start_utc, end_utc)
                if not stooq_df.empty:
                    price_df = stooq_df.copy()

        if price_df.empty:
            return pd.DataFrame()

        # Clean and normalize the downloaded data
        if isinstance(price_df.columns, pd.MultiIndex):
            price_df.columns = price_df.columns.get_level_values(0)
        price_df.columns = [str(col).capitalize() for col in price_df.columns]
        if "Close" not in price_df.columns and "Adj close" in price_df.columns:
            price_df = price_df.rename(columns={"Adj close": "Close"})

        if "Close" not in price_df.columns:
            return pd.DataFrame()

        price_df.index = pd.to_datetime(price_df.index, utc=True)
        price_df.index.name = "Date"
        
        # Convert all relevant columns to numeric, coercing errors to NaN
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in price_df.columns:
                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

        # Replace infinities with NaN
        price_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Ensure 'Volume', 'High', 'Low', 'Open' columns exist, fill with 'Close' or 0 if missing
        if "Volume" not in price_df.columns:
            price_df["Volume"] = 0
        if "High" not in price_df.columns:
            price_df["High"] = price_df["Close"]
        if "Low" not in price_df.columns:
            price_df["Low"] = price_df["Close"]
        if "Open" not in price_df.columns:
            price_df["Open"] = price_df["Close"]
            
       # print(f"DEBUG: In load_prices for {ticker}, 'Volume' column exists: {'Volume' in price_df.columns}, 'High' exists: {'High' in price_df.columns}, 'Low' exists: {'Low' in price_df.columns}, 'Open' exists: {'Open' in price_df.columns}") # Debug print
        
        price_df = price_df.dropna(subset=["Close"])
        price_df = price_df.ffill().bfill()

        # --- Save the entire fetched price data to cache ---
        if not price_df.empty:
            try:
                price_df.to_csv(cache_file)
            except Exception as e:
                print(f"⚠️ Could not write price cache file for {ticker}: {e}")
                
    # --- Fetch and merge financial data ---
    financial_df = pd.DataFrame()
    if financial_cache_file.exists():
        file_mod_time = datetime.fromtimestamp(financial_cache_file.stat().st_mtime, timezone.utc)
        if (datetime.now(timezone.utc) - file_mod_time) < timedelta(days=CACHE_DAYS * 4): # Financials update less frequently
            try:
                financial_df = pd.read_csv(financial_cache_file, index_col='Date', parse_dates=True)
                if financial_df.index.tzinfo is None:
                    financial_df.index = financial_df.index.tz_localize('UTC')
                else:
                    financial_df.index = financial_df.index.tz_convert('UTC')
            except Exception as e:
                print(f"⚠️ Could not read financial cache file for {ticker}: {e}. Refetching financials.")
    
    if financial_df.empty:
        if DATA_PROVIDER.lower() == 'alpaca':
            financial_df = _fetch_financial_data_from_alpaca(ticker)
            if financial_df.empty: # If Alpaca financial data is empty, fall back to Yahoo
                financial_df = _fetch_financial_data(ticker) # This calls the original Yahoo-based function
        else:
            financial_df = _fetch_financial_data(ticker) # This calls the original Yahoo-based function
            
        if not financial_df.empty:
            try:
                financial_df.to_csv(financial_cache_file)
            except Exception as e:
                print(f"⚠️ Could not write financial cache file for {ticker}: {e}")

    if not financial_df.empty and not price_df.empty:
        # Merge financial data by forward-filling the latest available financial report
        # Reindex financial_df to cover the full date range of price_df
        full_date_range = pd.date_range(start=price_df.index.min(), end=price_df.index.max(), freq='D', tz='UTC')
        financial_df_reindexed = financial_df.reindex(full_date_range)
        financial_df_reindexed = financial_df_reindexed.ffill()
        
        # Merge with price data
        final_df = price_df.merge(financial_df_reindexed, left_index=True, right_index=True, how='left')
        # Fill any remaining NaNs in financial features (e.g., at the very beginning)
        final_df.fillna(0, inplace=True)
    else:
        final_df = price_df.copy()

    # --- Add placeholder for sentiment data (for demonstration) ---
    # In a real scenario, this would be fetched from a sentiment API
    if 'Sentiment_Score' not in final_df.columns:
        final_df['Sentiment_Score'] = np.random.uniform(-1, 1, len(final_df)) # Placeholder: random sentiment
        final_df['Sentiment_Score'] = final_df['Sentiment_Score'].rolling(window=5).mean().fillna(0) # Smooth it a bit

    # Return the specifically requested slice
    return final_df.loc[(final_df.index >= _to_utc(start)) & (final_df.index <= _to_utc(end))].copy()

def _fetch_intermarket_data(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetches intermarket data (e.g., bond yields, commodities, currencies)."""
    intermarket_tickers = {
        '^VIX': 'VIX_Index',  # CBOE Volatility Index
        'DX-Y.NYB': 'DXY_Index', # U.S. Dollar Index
        'GC=F': 'Gold_Futures', # Gold Futures
        'CL=F': 'Oil_Futures',  # Crude Oil Futures
        '^TNX': 'US10Y_Yield',  # 10-Year Treasury Yield
        'USO': 'Oil_Price',    # United States Oil Fund ETF
        'GLD': 'Gold_Price',   # SPDR Gold Shares ETF
    }
    
    all_intermarket_dfs = []
    for ticker, name in intermarket_tickers.items():
        try:
            # Use load_prices_robust to respect DATA_PROVIDER and handle caching/fallbacks
            df = load_prices_robust(ticker, start, end)
            if not df.empty:
                # load_prices_robust returns a DataFrame with 'Close' column
                single_ticker_df = df[['Close']].rename(columns={'Close': name})
                all_intermarket_dfs.append(single_ticker_df)
        except Exception as e:
            print(f"  ⚠️ Could not fetch intermarket data for {ticker} ({name}): {e}")
            
    if not all_intermarket_dfs:
        return pd.DataFrame()

    # Concatenate all individual DataFrames
    intermarket_df = pd.concat(all_intermarket_dfs, axis=1)
    intermarket_df.index = pd.to_datetime(intermarket_df.index, utc=True)
    intermarket_df.index.name = "Date"
    
    # Calculate returns for intermarket features
    for col in intermarket_df.columns:
        intermarket_df[f"{col}_Returns"] = intermarket_df[col].pct_change(fill_method=None).fillna(0)
        
    return intermarket_df.ffill().bfill().fillna(0)


def fetch_training_data(ticker: str, data: pd.DataFrame, target_percentage: float = 0.05, class_horizon: int = CLASS_HORIZON) -> Tuple[pd.DataFrame, List[str]]:
    """Compute ML features from a given DataFrame."""
    print(f"  [DIAGNOSTIC] {ticker}: fetch_training_data - Initial data rows: {len(data)}")
    if data.empty or len(data) < FEAT_SMA_LONG + 10:
        print(f"  [DIAGNOSTIC] {ticker}: Skipping feature prep. Initial data has {len(data)} rows, required > {FEAT_SMA_LONG + 10}.")
        return pd.DataFrame(), []

    df = data.copy()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    # Ensure 'Close' is numeric and drop rows with NaN in 'Close'
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    
    if df.empty:
        print(f"  [DIAGNOSTIC] {ticker}: DataFrame became empty after dropping NaNs in 'Close'. Skipping feature prep.")
        return pd.DataFrame(), []

    # Fill missing values in other columns
    df = df.ffill().bfill()

    df = _calculate_technical_indicators(df) # Call the new function

    # --- Additional Financial Features (from _fetch_financial_data) ---
    financial_features = [col for col in df.columns if col.startswith('Fin_')]
    
    # Ensure these are numeric and fill NaNs if any remain
    for col in financial_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df["Target"]     = df["Close"].shift(-1)

    # Classification label for BUY model: class_horizon-day forward > +target_percentage
    fwd = df["Close"].shift(-class_horizon)
    df["TargetClassBuy"] = ((fwd / df["Close"] - 1.0) > target_percentage).astype(float)

    # Classification label for SELL model: class_horizon-day forward < -target_percentage
    df["TargetClassSell"] = ((fwd / df["Close"] - 1.0) < -target_percentage).astype(float)

    # Dynamically build the list of features that are actually present in the DataFrame
    # This is the most critical part to ensure consistency
    
    # Define a base set of expected technical features
    expected_technical_features = [
        "Close", "Volume", "High", "Low", "Open", "Returns", "SMA_F_S", "SMA_F_L", "Volatility", 
        "ATR", "RSI_feat", "MACD", "MACD_signal", "BB_upper", "BB_lower", "%K", "%D", "ADX",
        "OBV", "CMF", "ROC", "ROC_20", "ROC_60", "CMO", "KAMA", "EFI", "KC_Upper", "KC_Lower", "DC_Upper", "DC_Lower",
        "PSAR", "ADL", "CCI", "VWAP", "ATR_Pct", "Chaikin_Oscillator", "MFI", "OBV_SMA", "Historical_Volatility",
        "Market_Momentum_SPY",
        "Sentiment_Score",
        "VIX_Index_Returns", "DXY_Index_Returns", "Gold_Futures_Returns", "Oil_Futures_Returns", "US10Y_Yield_Returns",
        "Oil_Price_Returns", "Gold_Price_Returns"
    ]
    
    # Filter to only include technical features that are actually in df.columns
    present_technical_features = [col for col in expected_technical_features if col in df.columns]
    
    # Combine with financial features
    all_present_features = present_technical_features + financial_features
    
    # Also include target columns for the initial DataFrame selection before dropna
    target_cols = ["Target", "TargetClassBuy", "TargetClassSell"]
    cols_for_ready = all_present_features + target_cols
    
    # Filter cols_for_ready to ensure all are actually in df.columns (redundant but safe)
    cols_for_ready_final = [col for col in cols_for_ready if col in df.columns]

    ready = df[cols_for_ready_final].dropna()
    
    # The actual features used for training will be all columns in 'ready' except the target columns
    final_training_features = [col for col in ready.columns if col not in target_cols]

    print(f"   ↳ {ticker}: rows after features available: {len(ready)}")
    return ready, final_training_features

def load_prices_robust(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """A wrapper for load_prices that handles rate limiting with retries and other common API errors."""
    import time
    import random
    max_retries = 5
    base_wait_time = 5  # seconds, increased for more tolerance

    for attempt in range(max_retries):
        try:
            return load_prices(ticker, start, end)
        except Exception as e:
            error_str = str(e).lower()
            # Handle YFTzMissingError for delisted stocks gracefully
            if "yftzmissingerror" in error_str or "no timezone found" in error_str:
                print(f"  ℹ️ Skipping {ticker}: Data not available (possibly delisted).")
                return pd.DataFrame()
            
            # Handle rate limiting with exponential backoff
            if "yfratelimiterror" in error_str or "rate limit" in error_str or "429" in error_str:
                wait_time = base_wait_time * (2 ** attempt) + random.uniform(0, 1)
                print(f"  ⚠️ Rate limited trying to fetch {ticker}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                # For other unexpected errors, log it and fail for this ticker
                print(f"  ⚠️ An unexpected error occurred for {ticker}: {e}. Skipping.")
                return pd.DataFrame()
    
    print(f"  ❌ Failed to load data for {ticker} after {max_retries} retries due to persistent rate limiting.")
    return pd.DataFrame()
