import gym

# === PPO & Gym Integration ===
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces

# Extend your existing TradingEnv to be gym-compatible
class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.cash = 10000
        self.shares = 0
        self.initial_cash = 10000
        self.transaction_cost = 0.001
        self.done = False

        # Define action and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares = 0
        self.done = False
        return self._next_observation()

        
    def _next_observation(self):
        row = self.df.iloc[self.current_step]
        rsi = row.get('RSI', 0.0)
        hour = self.df.index[self.current_step].hour if hasattr(self.df.index[self.current_step], 'hour') else 0
        weekday = self.df.index[self.current_step].weekday() if hasattr(self.df.index[self.current_step], 'weekday') else 0
        obs = np.array([
            row.get('Close', 0.0),
            row.get('SMA_Short', 0.0),
            row.get('SMA_Long', 0.0),
            row.get('Return', 0.0),
            row.get('Volatility', 0.0),
            rsi,
            hour / 23.0,
            weekday / 6.0
        ], dtype=np.float32)
        return obs


    
    

        self.done = self.current_step >= len(self.df) - 1
        next_obs = self._next_observation()
        portfolio_value = self.cash + self.shares * price
        reward = portfolio_value - self.initial_cash

        return next_obs, reward, self.done, {}

# Train PPO agent
def train_ppo_agent(df):
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    return model


# === Original Code Below ===
import os
import sys
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from typing import List
from tqdm import tqdm
import time  # Import time module for delay
from sklearn.dummy import DummyClassifier  # Import a simple classifier for demonstration
from custom_strategy import CustomTradingStrategy  # Import the custom strategy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Add the src directory to the Python path
sys.path.append(os.path.dirname(__file__))

# Use an interactive backend for macOS
matplotlib.use("MacOSX")

# --- Constants ---
INITIAL_BALANCE = 20000
TRANSACTION_COST = 0.0015
POSITION_SIZE = 1.0  # Fixierte Positionsgröße auf 1
BACKTEST_PERIOD = 60
STOP_LOSS = 0.2  # Increased stop-loss threshold
TAKE_PROFIT = 0.2  # Increased take-profit threshold
MIN_HOLDING_PERIOD = 10000000  # Minimum holding period in steps
DEBUG_STEPS = False  # Debug switch for step method

# Define fixed stop-loss and take-profit thresholds
FIXED_STOP_LOSS = {}
FIXED_TAKE_PROFIT = {}

# --- Trading environment ---
class TradingEnv:
    """Custom trading environment for rule-based trading."""
    def __init__(self, df: pd.DataFrame, initial_balance: float = INITIAL_BALANCE, transaction_cost: float = TRANSACTION_COST):
        self.df = df.reset_index(drop=True)  # Ensure the DataFrame is reset and uses only the passed data
        print(f"Number of steps included in the backtest: {len(self.df)}")  # Print the number of steps
        self.current_step = 0
        self.cash = initial_balance
        self.shares = 0
        self.transaction_cost = transaction_cost
        self.portfolio_history = [initial_balance]
        self.trade_log = []
        self.returns = []
        self.stop_loss = STOP_LOSS  # Default stop-loss
        self.take_profit = TAKE_PROFIT  # Default take-profit
        self.dynamic_stop_loss = STOP_LOSS  # Dynamic stop-loss
        self.dynamic_take_profit = TAKE_PROFIT  # Dynamic take-profit
        self.trailing_stop_price = None  # Add a variable to track the trailing stop price
        self.holding_period = 0  # Track the holding period for shares

    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.cash = INITIAL_BALANCE
        self.shares = 0
        self.portfolio_history = [self.cash]
        self.trade_log = []
        self.returns = []
        self.trailing_stop_price = None  # Reset the trailing stop price
        self.holding_period = 0  # Reset holding period

    def detect_market_trend(self):
        """Detect market trend using moving averages."""
        short_window = 10  # Short-term moving average window
        long_window = 30  # Long-term moving average window

        # Calculate moving averages
        self.df['SMA_Short'] = self.df['Close'].rolling(window=short_window).mean()
        self.df['SMA_Long'] = self.df['Close'].rolling(window=long_window).mean()

        # Determine market trend
        if self.df['SMA_Short'].iloc[self.current_step] > self.df['SMA_Long'].iloc[self.current_step]:
            return "uptrend"
        elif self.df['SMA_Short'].iloc[self.current_step] < self.df['SMA_Long'].iloc[self.current_step]:
            return "downtrend"
        else:
            return "sideways"

    def step(self):
        """Execute one step with dynamic adjustments based on market trends."""
        current_price = float(self.df["Close"].iloc[self.current_step])
        previous_price = (
            float(self.df["Close"].iloc[self.current_step - 1])
            if self.current_step > 0
            else current_price
        )

        # Debug: Print current step, price, cash, and shares
        if DEBUG_STEPS:
            print(f"Step {self.current_step}: Current Price: {current_price:.2f}, Previous Price: {previous_price:.2f}, Cash: {self.cash:.2f}, Shares: {self.shares}")

        # Detect market trend
        market_trend = self.detect_market_trend()
        if DEBUG_STEPS:
            print(f"Step {self.current_step}: Market Trend: {market_trend}")

        # Debug: Print dynamic thresholds
        if DEBUG_STEPS:
            print(f"Step {self.current_step}: Adjusted Stop-Loss: {self.dynamic_stop_loss:.2%}, Adjusted Take-Profit: {self.dynamic_take_profit:.2%}")

        # Take Profit Logic
        if self.shares > 0:
            take_profit_price = self.trailing_stop_price / (1 - TAKE_PROFIT)
            if current_price >= take_profit_price:
                if DEBUG_STEPS:
                    print(f"Step {self.current_step}: Take Profit Triggered! Current Price: {current_price:.2f}, Take Profit Price: {take_profit_price:.2f}")
                self.cash += self.shares * current_price * (1 - self.transaction_cost)
                self.trade_log.append((self.current_step, "SELL", current_price, self.shares))
                self.shares = 0
                self.trailing_stop_price = None
                self.holding_period = 0  # Reset holding period after selling

        # Buy logic: Buy if price is increasing and no shares are held
        if self.shares == 0 and current_price > previous_price:
            max_shares = int((self.cash * POSITION_SIZE) / current_price)
            if max_shares > 0:
                self.shares = max_shares
                self.cash -= max_shares * current_price * (1 + self.transaction_cost)
                self.trade_log.append((self.current_step, "BUY", current_price, self.shares))
                self.trailing_stop_price = current_price * (1 - self.dynamic_stop_loss)
                if DEBUG_STEPS:
                    print(f"Step {self.current_step}: Bought {max_shares} shares at {current_price:.2f}")
                    print(f"Step {self.current_step}: Initial Trailing Stop Price: {self.trailing_stop_price:.2f}")

        # Update trailing stop-loss price if the price increases
        elif self.shares > 0:
            if current_price > self.trailing_stop_price / (1 - self.dynamic_stop_loss):
                self.trailing_stop_price = current_price * (1 - self.dynamic_stop_loss)
                if DEBUG_STEPS:
                    print(f"Step {self.current_step}: Updated Trailing Stop Price: {self.trailing_stop_price:.2f}")

            # Update holding period if shares are held
            self.holding_period += 1

            # Modify sell logic to enforce minimum holding period
            if self.holding_period >= MIN_HOLDING_PERIOD and current_price <= self.trailing_stop_price:
                if DEBUG_STEPS:
                    print(f"Step {self.current_step}: Trailing Stop-Loss Triggered! Current Price: {current_price:.2f}, Trailing Stop Price: {self.trailing_stop_price:.2f}")
                self.cash += self.shares * current_price * (1 - self.transaction_cost)
                self.trade_log.append((self.current_step, "SELL", current_price, self.shares))
                if DEBUG_STEPS:
                    print(f"Step {self.current_step}: Sold {self.shares} shares at {current_price:.2f} due to trailing stop-loss")
                self.shares = 0
                self.trailing_stop_price = None
                self.holding_period = 0  # Reset holding period after selling

        # Update portfolio value
        portfolio_value = self.cash + self.shares * current_price
        self.portfolio_history.append(portfolio_value)
        self.prev_portfolio_value = portfolio_value
        if DEBUG_STEPS:
            print(f"Step {self.current_step}: Portfolio value updated to {portfolio_value:.2f}")

        # Move to the next step
        self.current_step += 1

    def run(self):
        """Run the backtest."""
        while self.current_step < len(self.df) - 1:
            self.step()

    def render(self):
        """Render the portfolio value over time."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.portfolio_history)
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Steps")
        plt.ylabel("Portfolio Value ($)")
        plt.savefig(f"portfolio_{id(self)}.png")
        plt.show()
        plt.close()

# --- Predictive Model Functions ---
def train_predictive_model(df: pd.DataFrame):
    """
    Train a predictive model using the provided DataFrame.
    :param df: DataFrame containing features and target variable.
    :return: Trained model.
    """
    print(f"Initial dataset size: {len(df)} rows")
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    df['Returns'] = df['Close'].pct_change()
    print(f"Dataset size after calculating 'Returns': {len(df)} rows")
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    print(f"Dataset size after calculating 'SMA_10': {len(df)} rows")
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    print(f"Dataset size after calculating 'SMA_30': {len(df)} rows")
    df['Volatility'] = df['Close'].rolling(window=10).std()
    print(f"Dataset size after calculating 'Volatility': {len(df)} rows")

    X = df[['Close', 'Returns', 'SMA_10', 'SMA_30', 'Volatility']].values
    y = df['Target'].values

    print(f"Training dataset size: {len(X)} rows")
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # Increase estimators for better accuracy
    model.fit(X, y)  # Train on all rows
    return model

def predict_next_price(model, df):
    """Predict the next day's price."""
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    df['Returns'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()

    latest_data = df[['Returns', 'SMA_10', 'SMA_30', 'Volatility']].iloc[-1].values.reshape(1, -1)
    return model.predict(latest_data)[0]

# --- Parameter Optimization Function ---
def optimize_parameters(strategy, param_grid, X_train, y_train, cv=3, scoring='neg_mean_squared_error'):
    """
    Optimize parameters for the given strategy using GridSearchCV.
    :param strategy: The trading strategy to optimize.
    :param param_grid: Dictionary of parameters to search.
    :param X_train: Training features.
    :param y_train: Training target.
    :param cv: Number of cross-validation folds.
    :param scoring: Scoring metric for optimization.
    :return: Best parameters found by GridSearchCV.
    """
    grid_search = GridSearchCV(
        estimator=strategy,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,  # Use the scoring metric passed to the function
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

# --- Data fetching and preprocessing ---
def prepare_data(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True).dropna()
    if df.empty or len(df) < 20:
        raise ValueError(f"Insufficient data for {ticker} from {start} to {end}")
    return df

def calculate_volatility(df: pd.DataFrame) -> float:
    returns = df["Close"].pct_change().dropna()
    return returns.std().item()

def get_top_performing_stocks_ytd(sp500: bool = True, n: int = 10) -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    symbol_column = "Symbol"
    index_data = pd.read_html(url)[0]
    if symbol_column not in index_data.columns:
        raise KeyError(f"Column '{symbol_column}' not found in the table fetched from {url}")
    tickers = [symbol.replace('.', '-') for symbol in index_data[symbol_column].tolist()]
    performances = []
    end_date = datetime.today()
    start_date = datetime(end_date.year, 1, 1)
    for ticker in tqdm(tickers[:50], desc="Processing S&P 500 Tickers"):
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty or len(df) < 20 or "Close" not in df.columns:
            continue
        start_price = df["Close"].iloc[0].item()
        end_price = df["Close"].iloc[-1].item()
        growth = (end_price - start_price) / start_price
        performances.append((ticker, growth))
    top_tickers = [ticker for ticker, _ in sorted(performances, key=lambda x: x[1], reverse=True)[:n]]
    return top_tickers

def analyze_trades(trade_log: List[tuple]):
    buys = [trade for trade in trade_log if trade[1] == "BUY"]
    sells = [trade for trade in trade_log if trade[1] == "SELL"]
    completed_trades = min(len(buys), len(sells))
    profits = [sells[i][2] - buys[i][2] for i in range(completed_trades)]
    total_profit = sum(profits)
    win_rate = len([p for p in profits if p > 0]) / len(profits) if profits else 0
    avg_profit = total_profit / len(profits) if profits else 0
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Wins: {len([p for p in profits if p > 0])}, Losses: {len([p for p in profits if p <= 0])}")
    print(f"Win Rate: {win_rate:.2%}, Avg Profit: ${avg_profit:.2f}")

def analyze_trades_per_stock(trade_log: List[tuple], ticker: str, final_price: float):
    buys = [trade for trade in trade_log if trade[1] == "BUY"]
    sells = [trade for trade in trade_log if trade[1] == "SELL"]
    profits = [sells[i][2] - buys[i][2] for i in range(min(len(buys), len(sells)))]
    if len(buys) > len(sells):
        last_buy = buys[len(sells)]
        unrealized_profit = (final_price - last_buy[2]) * last_buy[3]
        profits.append(unrealized_profit)
        print(f"  Unrealized Profit for Open Position: ${unrealized_profit:.2f}")
    total_profit = sum(profits)
    win_rate = len([p for p in profits if p > 0]) / len(profits) if profits else 0
    print(f"\n📊 {ticker} Trade Analysis:")
    print(f"  Wins: {len([p for p in profits if p > 0])}, Losses: {len([p for p in profits if p <= 0])}")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Final Price: ${final_price:.2f}")

def calculate_weights(trade_logs: dict) -> dict:
    weights = {}
    total_profit = 0
    for ticker, log in trade_logs.items():
        buys = [trade for trade in log if trade[1] == "BUY"]
        sells = [trade for trade in log if trade[1] == "SELL"]
        profits = [sells[i][2] - buys[i][2] for i in range(min(len(buys), len(sells)))]
        stock_profit = sum(profits)
        weights[ticker] = max(stock_profit, 0)
        total_profit += max(stock_profit, 0)
    for ticker in weights:
        weights[ticker] = weights[ticker] / total_profit if total_profit > 0 else 1 / len(trade_logs)
    return weights

def pad_portfolio_history(portfolio_history: List[float], max_steps: int) -> List[float]:
    if len(portfolio_history) < max_steps:
        last_value = portfolio_history[-1] if portfolio_history else 0
        portfolio_history.extend([last_value] * (max_steps - len(portfolio_history)))
    return portfolio_history

def fetch_training_data(ticker: str) -> pd.DataFrame:
    """Fetch historical stock data for the last 30 days to account for rolling calculations."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=60)  # Fetch data for the last 30 days

    # Ensure end_date is not in the future
    if end_date > datetime.now():
        end_date = datetime.now()

    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True).dropna()
    if df.empty or len(df) < 30:  # Ensure at least 30 rows for rolling features
        print(f"⚠️ Insufficient data for {ticker} from {start_date} to {end_date}. Returning empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if insufficient data

    # Calculate additional features
    df['Returns'] = df['Close'].pct_change()  # Ensure 'Returns' is created first
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    df['Volatility'] = df['Returns'].rolling(window=10).std()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['Target'] = df['Close'].shift(-1)  # Predict the next day's price

    # Debug: Check if all required columns exist
    required_columns = ['Close', 'Returns', 'SMA_10', 'SMA_30', 'Volatility', 'Target']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"⚠️ Missing columns in DataFrame: {missing_columns}. Returning empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame if columns are missing

    print(f"✅ Fetched {len(df)} rows of data for {ticker} from {start_date} to {end_date}.")
    return df

# --- Main function ---
def main():
    """Main execution function."""
    os.makedirs("plots", exist_ok=True)
    print("🚀 Rule-Based Trading System")
    print("=" * 50 + "\n")
    print("🔍 Fetching top-performing stocks from S&P 500...")
    top_tickers = get_top_performing_stocks_ytd(sp500=True, n=5)
    print(f"📈 Selected tickers: {', '.join(top_tickers)}\n")

    models = {}  # Dictionary to store trained models for each stock

    for ticker in top_tickers:
        print(f"\n🔄 Fetching training data for {ticker}...")
        training_data = fetch_training_data(ticker).dropna()
        if training_data.empty:
            print(f"⚠️ Training data for {ticker} is empty. Skipping.")
            continue
        print(f"✅ Training data fetched with {len(training_data)} data points for {ticker}.\n")

        print(f"🔄 Training predictive model for {ticker}...")
        # Extract features and target for training
        X_train = training_data[['Close', 'Returns', 'SMA_10', 'SMA_30', 'Volatility']].values
        y_train = training_data['Target'].values

        model = train_predictive_model(training_data)  # Train the model for the current stock
        models[ticker] = model  # Store the trained model
        print(f"✅ Predictive model training completed for {ticker}.\n")

    if not models:
        print("⚠️ No models were trained. Exiting program.")
        return None, None  # Exit early if no models were trained

    print("🔄 Starting parameter optimization...")
    # Define the custom strategy
    strategy = CustomTradingStrategy()

    # Refine parameter grid
    param_grid = {
        'STOP_LOSS': [0.05, 0.1, 0.15],
        'TAKE_PROFIT': [0.05, 0.1, 0.15],
        # Entferne POSITION_SIZE aus der Optimierung
        'TRAILING_STOP': [0.01, 0.02, 0.03, 0.04, 0.05],  # Refined range for trailing stop
    }

    # Example: Use the first stock's data for parameter optimization
    first_ticker = next(iter(models.keys()))
    training_data = fetch_training_data(first_ticker)
    X_train = training_data[['Close', 'Returns', 'SMA_10', 'SMA_30', 'Volatility']].values
    y_train = training_data['Target'].values

    best_params = optimize_parameters(
        strategy, param_grid, X_train, y_train, cv=3, scoring='neg_mean_squared_error'  # Use regression scoring
    )
    print(f"🔧 Optimized Parameters: {best_params}")
    global STOP_LOSS, TAKE_PROFIT, TRAILING_STOP
    STOP_LOSS = best_params['STOP_LOSS']
    TAKE_PROFIT = best_params['TAKE_PROFIT']
    TRAILING_STOP = best_params['TRAILING_STOP']  # Update global variable for trailing stop

    print("✅ Parameter optimization completed.\n")

    start_date = datetime.today() - timedelta(days=BACKTEST_PERIOD + 365)  # Extend backtest period by 1 year
    end_date = datetime.today()

    # Initialize combined portfolio and individual stock contributions
    combined_portfolio = [0] * (BACKTEST_PERIOD + 365)
    buy_and_hold_portfolio = [0] * (BACKTEST_PERIOD + 365)
    individual_portfolios = {}
    trade_logs = {}

    # Allocate the initial balance equally across all selected stocks
    balance_per_stock = INITIAL_BALANCE / len(top_tickers)
    for ticker in top_tickers:
        print(f"\n📈 Fetching data for {ticker}...")
        df = prepare_data(ticker, start_date, end_date)
        stop_loss_threshold = min(max(calculate_volatility(df) * 2, 0.05), 0.5)
        print(f"🔍 Adjusted stop-loss for {ticker}: {stop_loss_threshold:.2%}")
        print(f"🔄 Running backtest for {ticker}...")
        env = TradingEnv(df, initial_balance=balance_per_stock)
        env.stop_loss = stop_loss_threshold
        env.run()
        print(f"✅ Backtest completed for {ticker}.")

        trade_logs[ticker] = env.trade_log
        individual_portfolios[ticker] = pad_portfolio_history(env.portfolio_history, BACKTEST_PERIOD + 365)

        # Analyze trades for the current stock
        print(f"🔍 Analyzing trades for {ticker}...")
        analyze_trades_per_stock(env.trade_log, ticker, final_price=float(df["Close"].iloc[-1]))
        print(f"✅ Trade analysis completed for {ticker}.")

        # Add the portfolio history to the combined portfolio
        for i in range(len(combined_portfolio)):
            combined_portfolio[i] += individual_portfolios[ticker][i]

        # Calculate buy-and-hold portfolio value for this stock
        initial_price = float(df["Close"].iloc[0])
        shares_held = balance_per_stock / initial_price
        stock_buy_and_hold_value = [float(shares_held * df["Close"].iloc[i]) for i in range(len(df))]
        stock_buy_and_hold_value = pad_portfolio_history(stock_buy_and_hold_value, BACKTEST_PERIOD + 365)
        buy_and_hold_portfolio = [
            buy_and_hold_portfolio[i] + stock_buy_and_hold_value[i]
            for i in range(len(buy_and_hold_portfolio))
        ]

    # Render the combined portfolio results
    print("\n📊 Rendering combined portfolio results...")
    plt.figure(figsize=(12, 6))
    for ticker, portfolio in individual_portfolios.items():
        plt.plot(portfolio, label=f"{ticker} Portfolio")
    plt.plot(combined_portfolio, label="Combined Portfolio", linewidth=2, color="black")
    plt.plot(buy_and_hold_portfolio, label="Buy-and-Hold Portfolio", linestyle="--", color="blue")

    # Highlight trailing stop-loss triggers and buy/sell actions
    for ticker, log in trade_logs.items():
        for step, action, price, shares in log:
            if action == "SELL":
                plt.scatter(step, combined_portfolio[step], color="red", label="Sell Action", zorder=5)
            elif action == "BUY":
                plt.scatter(step, combined_portfolio[step], color="green", label="Buy Action", zorder=5)

    # Avoid duplicate labels in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Combined Portfolio Value Over Time with Individual Contributions")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.ylim(0, max(max(combined_portfolio), max(buy_and_hold_portfolio)) * 1.1)
    plt.savefig("plots/combined_portfolio_with_individuals.png")
    plt.show()

    print("✅ Combined portfolio rendering completed.\n")
    return combined_portfolio, buy_and_hold_portfolio

if __name__ == "__main__":
    combined_portfolio, buy_and_hold_portfolio = main()
    final_combined_value = combined_portfolio[-1] if combined_portfolio else 0
    final_buy_and_hold_value = buy_and_hold_portfolio[-1] if buy_and_hold_portfolio else 0
    print(f"\n💰 Final Combined Portfolio Value: ${final_combined_value:.2f}")
    print(f"💰 Final Buy-and-Hold Portfolio Value: ${final_buy_and_hold_value:.2f}")
    # Plot the results
    if combined_portfolio and buy_and_hold_portfolio:
        plt.figure(figsize=(12, 6))
        plt.plot(combined_portfolio, label="Combined Portfolio", linewidth=2, color="black")
        plt.plot(buy_and_hold_portfolio, label="Buy-and-Hold Portfolio", linestyle="--", color="blue")
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Steps")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.savefig("plots/final_portfolio_comparison.png")
        plt.show()
# --- RSI Calculation ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))



    def determine_trend_signal(self, row):
        if row["MACD"] > row["MACD_signal"] and row["RSI"] < 70:
            return 1  # Aufwärtstrend
        elif row["MACD"] < row["MACD_signal"] and row["RSI"] > 30:
            return -1  # Abwärtstrend
        return 0  # neutral

    def determine_volatility_signal(self, row):
        if row["Volatility"] > self.df["Volatility"].quantile(0.7):
            return -1  # Risiko zu hoch -> Vorsicht
        elif row["Volatility"] < self.df["Volatility"].quantile(0.3):
            return 1  # Stabilität -> mehr investieren
        return 0

    def step(self, action):
        row = self.df.iloc[self.current_step]
        trend_signal = self.determine_trend_signal(row)
        volatility_signal = self.determine_volatility_signal(row)
        strategy_signal = trend_signal + volatility_signal

        position_fraction = np.clip(action[0], 0.0, 1.0)
        current_price = float(row["Close"])
        total_value = self.cash + self.shares * current_price
        target_shares = (position_fraction * total_value) / current_price
        delta_shares = target_shares - self.shares

        if strategy_signal > 0 and delta_shares > 0:
            buy_cost = delta_shares * current_price * (1 + self.transaction_cost)
            if self.cash >= buy_cost:
                self.cash -= buy_cost
                self.shares += delta_shares
                self.trade_log.append((self.current_step, "BUY", current_price, delta_shares))
        elif strategy_signal < 0 and delta_shares < 0:
            sell_shares = min(-delta_shares, self.shares)
            sell_value = sell_shares * current_price * (1 - self.transaction_cost)
            self.cash += sell_value
            self.shares -= sell_shares
            self.trade_log.append((self.current_step, "SELL", current_price, sell_shares))
        else:
            self.trade_log.append((self.current_step, "HOLD", current_price, 0))

        self.current_step += 1
