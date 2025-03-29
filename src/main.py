import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

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
        self.portfolio_history = []
        self.trade_log = []

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.cash = 10000
        self.shares = 0
        self.portfolio_history = []
        self.trade_log = []
        return self._next_observation()

    def _next_observation(self):
        obs = self.df.loc[self.current_step, ["Close", "RSI", "MACD", "MACD_signal", "Upper_Band", "Lower_Band"]].to_numpy()
        return obs.astype(np.float32)

    def step(self, action):
        action = int(action)
        current_price = float(self.df.loc[self.current_step, "Close"].item())

        if action == 1 and self.cash >= current_price:
            self.shares += 1
            self.cash -= current_price * (1 + self.transaction_cost)
            self.trade_log.append((self.current_step, "BUY", current_price))
        elif action == 2 and self.shares > 0:
            self.shares -= 1
            self.cash += current_price * (1 - self.transaction_cost)
            self.trade_log.append((self.current_step, "SELL", current_price))
        else:
            self.trade_log.append((self.current_step, "HOLD", current_price))

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_price = float(self.df.loc[self.current_step, "Close"].item())
        portfolio_value = self.cash + self.shares * next_price
        self.portfolio_history.append(portfolio_value)

        reward = (portfolio_value - (self.cash + self.shares * current_price)) / (self.cash + 1)
        return self._next_observation(), reward, done, {}


def get_top_performing_stocks(n=10):
    import datetime
    cache_file = "logs/top_tickers_cache.json"
    today = datetime.datetime.today().strftime("%Y-%m-%d")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)
            if cache.get("date") == today:
                return cache.get("tickers", [])

    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365)

    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = [symbol.replace('.', '-') for symbol in sp500['Symbol'].tolist()]

    performances = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty or len(df) < 2:
                continue
            start_price = float(df["Close"].iloc[0].item())
            end_price = float(df["Close"].iloc[-1].item())
            growth = (end_price - start_price) / start_price
            performances.append((ticker, growth))
        except Exception:
            continue

    sorted_performers = sorted(performances, key=lambda x: x[1], reverse=True)
    top_tickers = [ticker for ticker, _ in sorted_performers[:n]]

    with open(cache_file, "w") as f:
        json.dump({"date": today, "tickers": top_tickers}, f)

    return top_tickers

def analyze_trades(trade_log):
    trades = pd.DataFrame(trade_log, columns=["Step", "Action", "Price"])
    buy_prices = []
    profits = []

    for _, row in trades.iterrows():
        if row["Action"] == "BUY":
            buy_prices.append(row["Price"])
        elif row["Action"] == "SELL" and buy_prices:
            entry = buy_prices.pop(0)
            profit = row["Price"] - entry
            profits.append(profit)

    win_rate = np.mean([p > 0 for p in profits]) if profits else 0.0
    avg_profit = np.mean(profits) if profits else 0.0
    return win_rate, avg_profit

def main():
    tickers = get_top_performing_stocks()
    results = {}
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    for ticker in tickers:
        print(f"\n--- Training model for {ticker} ---")
        df = yf.download(ticker, period="1y", auto_adjust=True).dropna()
        prices = df["Close"]

        df['RSI'] = calculate_rsi(prices)
        df['MACD'], df['MACD_signal'] = calculate_macd(prices)
        df['Upper_Band'], df['Lower_Band'] = calculate_bollinger_bands(prices)
        df.dropna(inplace=True)

        env = TradingEnv(df)
        model_path = f"models/{ticker}_ppo.zip"

        if os.path.exists(model_path):
            print(f"Loading existing model for {ticker}...")
            model = PPO.load(model_path, env=env)
            model.set_env(env)
            model.learn(total_timesteps=25000)
        else:
            model = PPO("MlpPolicy", env, verbose=0)
            model.learn(total_timesteps=50000)

        model.save(model_path)

        obs = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        final_price = float(df["Close"].iloc[-1].item())
        final_portfolio_value = env.cash + env.shares * final_price

        print(f"Final Portfolio Value for {ticker}: ${final_portfolio_value:.2f}")
        print(f"Total Reward (Profit) for {ticker}: ${total_reward:.2f}")

        win_rate, avg_profit = analyze_trades(env.trade_log)
        print(f"Win Rate: {win_rate:.2%}, Avg Profit per Trade: ${avg_profit:.2f}")

        results[ticker] = env.portfolio_history

        # Save trade log to CSV
        trade_log_df = pd.DataFrame(env.trade_log, columns=["Step", "Action", "Price"])
        trade_log_df.to_csv(f"logs/{ticker}_trades.csv", index=False)

    plt.figure(figsize=(12, 6))
    for ticker, history in results.items():
        plt.plot(history, label=ticker)

    plt.title("Portfolio Value Over Time per Stock")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
