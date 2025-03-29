import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
from stable_baselines3 import PPO

# Technical indicators calculation functions
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    ema_short = prices.ewm(span=short_window, adjust=False).mean()
    ema_long = prices.ewm(span=long_window, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(prices, window=20):
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

# Custom Gym environment for trading
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.cash = 10000
        self.shares = 0
        self.transaction_cost = 0.01

        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observations: price, RSI, MACD, MACD signal, upper_band, lower_band
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.cash = 10000
        self.shares = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.loc[self.current_step, ["Close", "RSI", "MACD", "MACD_signal", "Upper_Band", "Lower_Band"]].to_numpy()
        return obs.astype(np.float32)

    def step(self, action):
        action = int(action)  # Ensure scalar for comparison
        current_price = float(self.df.loc[self.current_step, "Close"].item())

        # Execute trades
        if action == 1 and self.cash >= current_price:
            self.shares += 1
            self.cash -= current_price * (1 + self.transaction_cost)
        elif action == 2 and self.shares > 0:
            self.shares -= 1
            self.cash += current_price * (1 - self.transaction_cost)

        self.current_step += 1

        done = self.current_step >= len(self.df) - 1
        next_price = float(self.df.loc[self.current_step, "Close"].item())
        portfolio_value = self.cash + self.shares * next_price

        reward = portfolio_value - (self.cash + self.shares * current_price)

        return self._next_observation(), reward, done, {}


def main():
    # Fetch historical stock data
    df = yf.download("AAPL", period="1y", auto_adjust=True).dropna()
    prices = df["Close"]

    # Calculate technical indicators
    df['RSI'] = calculate_rsi(prices)
    df['MACD'], df['MACD_signal'] = calculate_macd(prices)
    df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(prices)
    df.dropna(inplace=True)

    # Ensure the DataFrame is not empty
    if df.empty:
        raise ValueError("The dataset is empty after processing. Ensure valid data is available.")

    # Initialize the Gym environment
    env = TradingEnv(df)

    # Train PPO agent
    print("Training PPO Reinforcement Learning model...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Evaluate the trained agent
    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    # Safely access the last close price
    last_close_price = df['Close'].iloc[-1] if 'Close' in df.columns and not df.empty else None
    if last_close_price is None:
        raise ValueError("Unable to access the last close price. Check the DataFrame structure.")

    # Calculate and print the final portfolio value
    final_portfolio_value = env.cash + env.shares * last_close_price
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    print(f"Total Reward (Profit): ${total_reward:.2f}")

if __name__ == "__main__":
    main()
