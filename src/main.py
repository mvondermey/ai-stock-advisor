import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU usage
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
import json
import numpy as np  # Correct the NumPy import statement
import pandas as pd
import yfinance as yf
import gymnasium as gym  # Replace gym with gymnasium
from gymnasium import spaces  # Update import for spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import multiprocessing
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
# warnings.filterwarnings('ignore')

# --- Constants ---
INITIAL_BALANCE = 10000
TRANSACTION_COST = 0.001  # Reduced from 0.01
POSITION_SIZE = 0.1  # Risk 10% per trade
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.04  # 4% take profit
TRAINING_TIMESTEPS = 100000  # Increased from 50000
BACKTEST_DAYS = 90  # Days for backtesting


# --- Technical indicators ---
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI) using Wilder's smoothing."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, 
                 short_window: int = 12, 
                 long_window: int = 26, 
                 signal_window: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD and Signal line."""
    ema_short = prices.ewm(span=short_window, adjust=False).mean()
    ema_long = prices.ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    return sma + (std * 2), sma - (std * 2)

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate Average True Range (ATR)."""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

# --- Trading environment ---
class TradingEnv(gym.Env):  # Ensure compatibility with gymnasium.Env
    """Custom trading environment for RL with enhanced features."""
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                df: pd.DataFrame, 
                initial_balance: float = INITIAL_BALANCE, 
                transaction_cost: float = TRANSACTION_COST,
                position_size: float = POSITION_SIZE,
                stop_loss_pct: float = STOP_LOSS_PCT,
                take_profit_pct: float = TAKE_PROFIT_PCT):
        
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.cash = initial_balance
        self.shares = 0
        self.transaction_cost = transaction_cost
        self.portfolio_history = [initial_balance]
        self.trade_log = []
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.stop_loss = None
        self.take_profit = None
        self.max_drawdown = 0
        self.peak_value = initial_balance

        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 12 features (expanded from 6)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(12,), 
            dtype=np.float32
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)  # Set the random seed if provided
        self.current_step = 0
        self.cash = self.portfolio_history[0]
        self.shares = 0
        self.portfolio_history = [self.cash]
        self.trade_log = []
        self.stop_loss = None
        self.take_profit = None
        self.max_drawdown = 0
        self.peak_value = self.cash
        observation = self._next_observation()
        
        # Return only observation if used with SubprocVecEnv
        if isinstance(self, SubprocVecEnv):
            return observation
        return observation, {}  # Return observation and an empty info dictionary

    def _next_observation(self) -> np.ndarray:
        """Get the next observation with expanded features."""
        try:
            close_prices = self.df.loc[max(0, self.current_step-5):self.current_step, "Close"]
            z_score = (self.df.loc[self.current_step, "Close"] - close_prices.mean()) / (close_prices.std() + 1e-10)
            
            obs = np.array([
                float(np.nan_to_num(self.df.loc[self.current_step, "Close"].iloc[0], nan=0.0)),
                float(np.nan_to_num(self.df.loc[self.current_step, "RSI"].iloc[0], nan=0.0)),
                float(np.nan_to_num(self.df.loc[self.current_step, "MACD"].iloc[0], nan=0.0)),
                float(np.nan_to_num(self.df.loc[self.current_step, "MACD_signal"].iloc[0], nan=0.0)),
                float(np.nan_to_num(self.df.loc[self.current_step, "Upper_Band"].iloc[0], nan=0.0)),
                float(np.nan_to_num(self.df.loc[self.current_step, "Lower_Band"].iloc[0], nan=0.0)),
                float(self.shares),  # Current position
                float(self.cash / INITIAL_BALANCE),  # Normalized cash
                float(np.nan_to_num(self.df.loc[self.current_step, "Volume"].iloc[0] / 1e6, nan=0.0)),  # Normalized volume
                float(np.nan_to_num(self.df.loc[self.current_step, "ATR"].iloc[0], nan=0.0)),  # Volatility measure
                float(np.nan_to_num(z_score, nan=0.0)),  # Z-score, replace NaN with 0
                float(self.max_drawdown)  # Risk metric
            ], dtype=np.float32)
            
            # Log if any NaN values are detected
            if np.any(np.isnan(obs)):
                print(f"⚠️ NaN detected in observation at step {self.current_step}: {obs}")
            
            return obs
        except Exception as e:
            print(f"❌ Error in _next_observation at step {self.current_step}: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)  # Return a safe default

    def _execute_trade(self, action: int, current_price: float):
        """Execute trade with position sizing and risk management."""
        if action == 1:  # BUY
            if self.shares == 0:  # Only enter new position if none exists
                max_shares = int((self.cash * self.position_size) / current_price)
                if max_shares > 0:
                    self.shares = max_shares
                    self.cash -= max_shares * current_price * (1 + self.transaction_cost)
                    self.stop_loss = current_price * (1 - self.stop_loss_pct)
                    self.take_profit = current_price * (1 + self.take_profit_pct)
                    self.trade_log.append((self.current_step, "BUY", current_price, self.shares))
                    
        elif action == 2 and self.shares > 0:  # SELL
            self.cash += self.shares * current_price * (1 - self.transaction_cost)
            self.trade_log.append((self.current_step, "SELL", current_price, self.shares))
            self.shares = 0
            self.stop_loss = None
            self.take_profit = None

def step(self, action: int):
    """Execute one step in the environment."""
    try:
        action = int(action)
        current_price = float(np.nan_to_num(self.df.loc[self.current_step, "Close"].item(), nan=0.0))
        
        # Execute trade based on action
        self._execute_trade(action, current_price)
        
        # Check stop loss/take profit
        if self.shares > 0:
            if current_price <= self.stop_loss:
                self.cash += self.shares * current_price * (1 - self.transaction_cost)
                self.trade_log.append((self.current_step, "SL", current_price, self.shares))
                self.shares = 0
                self.stop_loss = None
                self.take_profit = None
            elif current_price >= self.take_profit:
                self.cash += self.shares * current_price * (1 - self.transaction_cost)
                self.trade_log.append((self.current_step, "TP", current_price, self.shares))
                self.shares = 0
                self.stop_loss = None
                self.take_profit = None
        else:
            self.trade_log.append((self.current_step, "HOLD", current_price, 0))

        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Calculate portfolio value
        next_price = float(np.nan_to_num(self.df.loc[self.current_step, "Close"].item(), nan=0.0))
        portfolio_value = self.cash + self.shares * next_price
        
        if not isinstance(self.portfolio_history, list):
            self.portfolio_history = list(self.portfolio_history)
        self.portfolio_history.append(portfolio_value)
        
        # Update max drawdown
        self.peak_value = max(self.peak_value, portfolio_value)
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate reward with risk adjustment
        if len(self.portfolio_history) > 1:
            daily_return = (portfolio_value - self.portfolio_history[-2]) / (self.portfolio_history[-2] + 1e-10)
        else:
            daily_return = 0.0
        
        recent_history = self.portfolio_history[-10:]
        if len(recent_history) > 1:
            denom = np.array(self.portfolio_history[-10:-1]) + 1e-10  # 🛠️ Fix applied here
            volatility = np.std(np.diff(recent_history) / denom) + 1e-10
        else:
            volatility = 1e-10
        
        reward = np.nan_to_num(daily_return / volatility - 2 * current_drawdown, nan=0.0)
        
        if np.isnan(reward):
            print(f"⚠️ NaN detected in reward at step {self.current_step}: daily_return={daily_return}, volatility={volatility}, current_drawdown={current_drawdown}")
        
        return self._next_observation(), reward, terminated, truncated, {}
    
    except Exception as e:
        print(f"❌ Error in step at step {self.current_step}: {e}")
        return self._next_observation(), 0.0, True, False, {"error": str(e)}


    def render(self, mode='human'):
        """Render the current state (for monitoring)."""
        if mode == 'human':
            current_price = self.df.loc[self.current_step, "Close"]
            print(f"Step: {self.current_step}, Price: {current_price:.2f}, "
                 f"Shares: {self.shares}, Cash: {self.cash:.2f}, "
                 f"Portfolio Value: {self.portfolio_history[-1]:.2f}")
        else:
            super().render(mode=mode)  # Ensure compatibility with gymnasium

# --- Analytics ---
def analyze_trades(trade_log: list, prices: pd.Series) -> dict:
    """Enhanced trade analysis with more metrics."""
    trades = pd.DataFrame(trade_log, columns=["Step", "Action", "Price", "Shares"])
    trades['Date'] = prices.index[trades['Step']]
    trades['Value'] = trades['Price'] * trades['Shares']
    
    results = {
        'num_trades': len(trades[trades['Action'].isin(['BUY', 'SELL', 'SL', 'TP'])]),
        'win_rate': 0.0,
        'avg_profit': 0.0,
        'profit_factor': 0.0,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'avg_trade_duration': 0
    }
    
    if len(trades) < 2:
        return results
    
    # Calculate returns for each trade
    positions = []
    trade_returns = []
    trade_durations = []
    
    for i, row in trades.iterrows():
        if row['Action'] == 'BUY':
            positions.append({'entry_step': row['Step'], 
                             'entry_price': row['Price'],
                             'entry_date': row['Date']})
        elif row['Action'] in ['SELL', 'SL', 'TP'] and positions:
            position = positions.pop(0)
            duration = (row['Date'] - position['entry_date']).days
            trade_return = (row['Price'] - position['entry_price']) / position['entry_price']
            
            trade_returns.append(trade_return)
            trade_durations.append(duration)
    
    if trade_returns:
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        
        results.update({
            'win_rate': len(wins) / len(trade_returns),
            'avg_profit': np.mean(trade_returns),
            'profit_factor': sum(wins) / (-sum(losses)) if losses else float('inf'),
            'max_drawdown': min(trade_returns),
            'sharpe_ratio': (np.mean(trade_returns) / (np.std(trade_returns) + 1e-10)) * np.sqrt(252),
            'sortino_ratio': (np.mean(trade_returns) / (np.std(losses) + 1e-10)) * np.sqrt(252),
            'avg_trade_duration': np.mean(trade_durations) if trade_durations else 0
        })
    
    return results

# --- Data fetching and preprocessing ---
def get_top_performing_stocks(n: int = 10) -> List[str]:
    """Get top n performing S&P 500 stocks with caching."""
    cache_file = "logs/top_tickers_cache.json"
    today = datetime.today().strftime("%Y-%m-%d")
    
    # Try to load from cache
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
            if cache.get("date") == today:
                return cache.get("tickers", [])[:n]
    
    # Fetch fresh data
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = [symbol.replace('.', '-') for symbol in sp500['Symbol'].tolist()]
    
    performances = []
    end = datetime.today()
    start = end - timedelta(days=365)
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(get_stock_performance, ticker, start, end): ticker for ticker in tickers}
        for future in tqdm(as_completed(futures), total=len(tickers), desc="Evaluating S&P 500 stocks"):
            ticker = futures[future]
            try:
                result = future.result()
                if result:
                    performances.append(result)
            except Exception as e:
                print(f"❌ Failed to process {ticker}: {e}")
    
    top_tickers = [t for t, _ in sorted(performances, key=lambda x: x[1], reverse=True)[:n]]
    
    # Update cache
    os.makedirs("logs", exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({"date": today, "tickers": top_tickers}, f)
    
    return top_tickers

def get_stock_performance(ticker: str, start: datetime, end: datetime) -> Optional[Tuple[str, float]]:
    """Helper function to get single stock performance."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty or len(df) < 20:  # Minimum 20 days of data
            return None
        
        start_price = float(df["Close"].iloc[0].item())
        end_price = float(df["Close"].iloc[-1].item())
        growth = (end_price - start_price) / start_price
        return (ticker, growth)
    except Exception:
        return None

def prepare_data(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download and preprocess data for a given ticker and date range."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True).dropna()
    
    if df.empty:
        raise ValueError(f"No data available for {ticker} from {start} to {end}")
    
    # Calculate technical indicators
    prices = df["Close"]
    df['RSI'] = calculate_rsi(prices)
    df['MACD'], df['MACD_signal'] = calculate_macd(prices)
    df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(prices)
    df['ATR'] = calculate_atr(df)
    df['OBV'] = calculate_obv(df)
    
    # Drop any remaining NA values
    df.dropna(inplace=True)
    
    return df

# --- Model training and evaluation ---
class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional values to TensorBoard."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.portfolio_values = []
        
    def _on_step(self) -> bool:
        # Log portfolio value using get_attr for SubprocVecEnv
        portfolio_values = self.training_env.get_attr("portfolio_history")
        for value in portfolio_values:
            if value:
                self.portfolio_values.append(value[-1])
                self.logger.record('portfolio/value', value[-1])
        
        return True

def train_model(env: gym.Env, 
               ticker: str, 
               timesteps: int = TRAINING_TIMESTEPS,
               learning_rate: float = 3e-4,
               tensorboard_log: Optional[str] = None) -> PPO:
    """Train a PPO model on the given environment."""
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        device="cpu",
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=tensorboard_log
    )
    
    # Create callbacks
    callbacks = [TensorboardCallback()]
    
    # Train the model
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    return model

def evaluate_model(model: PPO, 
                 env: gym.Env, 
                 render: bool = False) -> dict:
    """Evaluate a trained model on the given environment."""
    obs = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        
        if render:
            env.render()
    
    # Get final metrics
    final_value = env.get_attr("cash")[0] + env.get_attr("shares")[0] * env.get_attr("df")[0]["Close"].iloc[-1]
    trade_log = env.get_attr("trade_log")[0]
    prices = env.get_attr("df")[0]["Close"]
    metrics = analyze_trades(trade_log, prices)
    metrics['final_value'] = final_value
    
    return metrics

# --- Pipeline functions ---
def run_ticker_pipeline(ticker: str, 
                       timeframes: List[int] = [0, 90, 180]) -> List[dict]:
    """Complete pipeline for a single ticker across multiple timeframes."""
    start_time = time.time()
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    base_model_path = os.path.join("models", f"{ticker}_base_model.zip")  # Ensure correct file extension
    analytics_summary = []
    
    # Train on each timeframe
    for offset in timeframes:
        end_train = datetime.today() - timedelta(days=offset)
        start_train = end_train - timedelta(days=365)
        
        try:
            # Prepare data
            df_train = prepare_data(ticker, start_train, end_train)
            
            # Create vectorized environment
            num_envs = max(multiprocessing.cpu_count() // 2, 1)  # Use half the cores
            env = SubprocVecEnv([lambda: TradingEnv(df_train) for _ in range(num_envs)])
            
            # Train or load model
            if offset == timeframes[0]:  # First timeframe - train new model
                print(f"\n🚀 Training new model for {ticker} (offset: {offset} days)")
                model = train_model(
                    env, 
                    ticker,
                    tensorboard_log=f"logs/{ticker}"
                )
                model.save(base_model_path)  # Save model with correct path
            else:  # Subsequent timeframes - fine-tune existing model
                print(f"\n🔄 Fine-tuning model for {ticker} (offset: {offset} days)")
                if os.path.exists(base_model_path):
                    model = PPO.load(base_model_path, env=env, device="cpu")  # Load model with correct path
                else:
                    print(f"❌ Model file not found for {ticker} (offset {offset}). Skipping...")
                    continue  # Skip this offset if the model file is missing
                model.set_env(env)
                model.learn(total_timesteps=TRAINING_TIMESTEPS//2)  # Half the timesteps for fine-tuning
            
            # Evaluate
            metrics = evaluate_model(model, env)
            metrics.update({
                "Ticker": ticker,
                "Offset Days": offset,
                "Phase": "Training"  # Add Phase key for training
            })
            analytics_summary.append(metrics)
            
        except Exception as e:
            print(f"❌ Failed to process {ticker} (offset {offset}): {e}")
            continue
    
    # Backtest on most recent data
    try:
        print(f"\n🧪 Backtesting {ticker} on most recent {BACKTEST_DAYS} days")
        end_bt = datetime.today()
        start_bt = end_bt - timedelta(days=BACKTEST_DAYS)
        df_bt = prepare_data(ticker, start_bt, end_bt)
        
        # Create backtest environment
        env_bt = DummyVecEnv([lambda: TradingEnv(df_bt)])  # Single env for backtest
        
        # Load best model
        if os.path.exists(base_model_path):
            model = PPO.load(base_model_path, env=env_bt, device="cpu")  # Load model with correct path
            model.set_env(env_bt)
        else:
            print(f"❌ Model file not found for {ticker} during backtest. Skipping...")
            return analytics_summary  # Skip backtest if the model file is missing
        
        # Evaluate
        metrics = evaluate_model(model, env_bt, render=True)
        metrics.update({
            "Ticker": ticker,
            "Offset Days": 999,  # Special code for backtest
            "Phase": "Backtest"  # Add Phase key for backtesting
        })
        analytics_summary.append(metrics)
        
    except Exception as e:
        print(f"❌ Failed to backtest {ticker}: {e}")
    
    # Print summary
    elapsed = time.time() - start_time
    print(f"\n✅ Completed {ticker} in {elapsed:.2f} seconds")
    
    return analytics_summary

# --- Visualization ---
def plot_results(df_summary: pd.DataFrame):
    """Generate and save performance plots."""
    os.makedirs("logs/plots", exist_ok=True)
    
    # Portfolio Value Comparison
    plt.figure(figsize=(14, 7))
    for ticker in df_summary['Ticker'].unique():
        subset = df_summary[df_summary['Ticker'] == ticker]
        training = subset[subset['Phase'] == 'Training'].sort_values('Offset Days')
        backtest = subset[subset['Phase'] == 'Backtest']
        
        # Plot training performance
        plt.plot(training['Offset Days'], training['final_value'], 
                'o-', label=f"{ticker} (Training)")
        
        # Plot backtest performance
        if not backtest.empty:
            plt.plot([999], backtest['final_value'], 
                    '^', markersize=10, label=f"{ticker} (Backtest)")
    
    plt.title("Portfolio Value Across Training Periods and Backtest")
    plt.xlabel("Training End Date Offset (Days from Today)")
    plt.ylabel("Final Portfolio Value ($)")
    plt.xticks([0, 90, 180, 999], ['0 (Recent)', '90', '180', 'Backtest'])
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("logs/plots/portfolio_value_comparison.png", bbox_inches='tight')
    plt.show()
    
    # Risk-Return Scatter Plot
    plt.figure(figsize=(12, 6))
    for ticker in df_summary['Ticker'].unique():
        subset = df_summary[df_summary['Ticker'] == ticker]
        x = subset['sharpe_ratio']
        y = subset['final_value'] / INITIAL_BALANCE - 1  # Return
        s = subset['win_rate'] * 100  # Marker size
        
        plt.scatter(x, y, s=s, alpha=0.6, label=ticker)
        plt.annotate(ticker, (x.iloc[-1], y.iloc[-1]), textcoords="offset points", 
                    xytext=(0,5), ha='center', fontsize=8)
    
    plt.title("Risk-Return Profile (Size = Win Rate %)")
    plt.xlabel("Sharpe Ratio")
    plt.ylabel("Total Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/plots/risk_return_profile.png")
    plt.show()

# --- Main function ---
def main():
    """Main execution function."""
    print("\n" + "="*50)
    print("🚀 Trading AI System - Enhanced Version")
    print("="*50 + "\n")
    
    # System info
    print(f"✅ Using device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print("✅ Using CPU (no CUDA device available)")
    print(f"📊 Available CPU cores: {multiprocessing.cpu_count()}\n")
    
    # Get top performing stocks
    print("🔍 Identifying top performing S&P 500 stocks...")
    tickers = get_top_performing_stocks(n=5)  # Reduced to 5 for demo
    print(f"📈 Top tickers selected: {', '.join(tickers)}\n")
    
    # Run pipeline for each ticker
    analytics_summary = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_ticker_pipeline, ticker): ticker for ticker in tickers}
        
        for future in tqdm(as_completed(futures), total=len(tickers), desc="Processing Tickers"):
            ticker = futures[future]
            try:
                result = future.result()
                analytics_summary.extend(result)
            except Exception as e:
                print(f"❌ Failed to process {ticker}: {e}")
    
    # Debug: Print analytics_summary to verify structure
    print(f"🔍 Debug: analytics_summary = {analytics_summary}")
    
    # Save and analyze results
    df_summary = pd.DataFrame(analytics_summary)
    df_summary.to_csv("logs/trade_analytics_summary.csv", index=False)
    
    # Print top performers
    if 'Phase' in df_summary.columns:
        top_performers = df_summary[df_summary['Phase'] == 'Backtest'].sort_values(
            'final_value', ascending=False)
        print("\n🏆 Top Performing Models (Backtest):")
        print(top_performers[['Ticker', 'final_value', 'win_rate', 'sharpe_ratio']].head())
    else:
        print("❌ 'Phase' column missing in df_summary. Check analytics_summary structure.")
    
    # Generate plots
    plot_results(df_summary)
    
    print("\n✅ All tasks completed successfully!")

if __name__ == "__main__":
    main()


