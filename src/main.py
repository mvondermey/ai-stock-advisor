import os
import sys
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from typing import List
from tqdm import tqdm
from parameter_optimization import optimize_parameters
from predictive_model import train_predictive_model, predict_next_price
from reinforcement_learning import TradingEnvRL
import time  # Import time module for delay
from sklearn.dummy import DummyClassifier  # Import a simple classifier for demonstration
from custom_strategy import CustomTradingStrategy  # Import the custom strategy
# Add the src directory to the Python path
sys.path.append(os.path.dirname(__file__))

# Use an interactive backend for macOS
matplotlib.use("MacOSX")

# --- Constants ---
INITIAL_BALANCE = 20000
TRANSACTION_COST = 0.0015
POSITION_SIZE = 0.2  # Reduced position size for more conservative trading
BACKTEST_PERIOD = 60
STOP_LOSS = 0.2  # Increased stop-loss threshold
TAKE_PROFIT = 0.2  # Increased take-profit threshold
MIN_HOLDING_PERIOD = 5  # Minimum holding period in steps

# Define fixed stop-loss and take-profit thresholds
FIXED_STOP_LOSS = {}
FIXED_TAKE_PROFIT = {}

# --- Trading environment ---
class TradingEnv:
    """Custom trading environment for rule-based trading."""
    def __init__(self, df: pd.DataFrame, initial_balance: float = INITIAL_BALANCE, transaction_cost: float = TRANSACTION_COST):
        self.df = df.reset_index(drop=True)
        self.current_step = 0
        self.cash = initial_balance
        self.shares = 0
        self.transaction_cost = transaction_cost
        self.portfolio_history = [initial_balance]
        self.trade_log = []
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
        print(f"Step {self.current_step}: Current Price: {current_price:.2f}, Previous Price: {previous_price:.2f}, Cash: {self.cash:.2f}, Shares: {self.shares}")

        # Detect market trend
        market_trend = self.detect_market_trend()
        print(f"Step {self.current_step}: Market Trend: {market_trend}")

        # Adjust strategy dynamically based on market trend
        if market_trend == "uptrend":
            self.dynamic_stop_loss = STOP_LOSS * 0.5  # Tighten stop-loss in uptrend
            self.dynamic_take_profit = TAKE_PROFIT * 2  # Increase take-profit in uptrend
        elif market_trend == "downtrend":
            self.dynamic_stop_loss = STOP_LOSS * 2  # Widen stop-loss in downtrend
            self.dynamic_take_profit = TAKE_PROFIT * 0.5  # Reduce take-profit in downtrend
        else:
            self.dynamic_stop_loss = STOP_LOSS  # Default stop-loss
            self.dynamic_take_profit = TAKE_PROFIT  # Default take-profit

        # Debug: Print dynamic thresholds
        print(f"Step {self.current_step}: Adjusted Stop-Loss: {self.dynamic_stop_loss:.2%}, Adjusted Take-Profit: {self.dynamic_take_profit:.2%}")

        # Buy logic: Buy if price is increasing and no shares are held
        if self.shares == 0 and current_price > previous_price:
            max_shares = int((self.cash * POSITION_SIZE) / current_price)
            if max_shares > 0:
                self.shares = max_shares
                self.cash -= max_shares * current_price * (1 + self.transaction_cost)
                self.trade_log.append((self.current_step, "BUY", current_price, self.shares))
                self.trailing_stop_price = current_price * (1 - self.dynamic_stop_loss)
                print(f"Step {self.current_step}: Bought {max_shares} shares at {current_price:.2f}")
                print(f"Step {self.current_step}: Initial Trailing Stop Price: {self.trailing_stop_price:.2f}")

        # Update trailing stop-loss price if the price increases
        elif self.shares > 0:
            if current_price > self.trailing_stop_price / (1 - self.dynamic_stop_loss):
                self.trailing_stop_price = current_price * (1 - self.dynamic_stop_loss)
                print(f"Step {self.current_step}: Updated Trailing Stop Price: {self.trailing_stop_price:.2f}")

            # Update holding period if shares are held
            self.holding_period += 1

            # Modify sell logic to enforce minimum holding period
            if self.holding_period >= MIN_HOLDING_PERIOD and current_price <= self.trailing_stop_price:
                print(f"Step {self.current_step}: Trailing Stop-Loss Triggered! Current Price: {current_price:.2f}, Trailing Stop Price: {self.trailing_stop_price:.2f}")
                self.cash += self.shares * current_price * (1 - self.transaction_cost)
                self.trade_log.append((self.current_step, "SELL", current_price, self.shares))
                print(f"Step {self.current_step}: Sold {self.shares} shares at {current_price:.2f} due to trailing stop-loss")
                self.shares = 0
                self.trailing_stop_price = None
                self.holding_period = 0  # Reset holding period after selling

        # Update portfolio value
        portfolio_value = self.cash + self.shares * current_price
        self.portfolio_history.append(portfolio_value)
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

# --- Main function ---
def main():
    """Main execution function."""
    os.makedirs("plots", exist_ok=True)
    print("🚀 Rule-Based Trading System")
    print("=" * 50 + "\n")
    print("🔍 Fetching top-performing stocks from S&P 500...")
    top_tickers = get_top_performing_stocks_ytd(sp500=True, n=5)
    print(f"📈 Selected tickers: {', '.join(top_tickers)}\n")

    # Expand training data
    X_train = [[0.01, 0.02], [0.02, 0.03], [0.03, 0.04], [0.04, 0.05], [0.05, 0.06], [0.06, 0.07]]
    y_train = [1, 0, 1, 0, 1, 0]  # Ensure balanced classes with more samples

    # Define the custom strategy
    strategy = CustomTradingStrategy()

    # Refine parameter grid
    param_grid = {
        'STOP_LOSS': [0.05, 0.1, 0.15],
        'TAKE_PROFIT': [0.05, 0.1, 0.15],
        'POSITION_SIZE': [0.1, 0.15, 0.2],  # Narrowed range for POSITION_SIZE
        'TRAILING_STOP': [0.01, 0.02, 0.03, 0.04, 0.05],  # Refined range for trailing stop
    }

    # Add cross-validation during optimization
    best_params = optimize_parameters(strategy, param_grid, X_train, y_train, cv=3)  # Use 3-fold cross-validation
    print(f"🔧 Optimized Parameters: {best_params}")
    global STOP_LOSS, TAKE_PROFIT, POSITION_SIZE, TRAILING_STOP
    STOP_LOSS = best_params['STOP_LOSS']
    TAKE_PROFIT = best_params['TAKE_PROFIT']
    POSITION_SIZE = best_params['POSITION_SIZE']
    TRAILING_STOP = best_params['TRAILING_STOP']  # Update global variable for trailing stop

    start_date = datetime.today() - timedelta(days=BACKTEST_PERIOD)
    end_date = datetime.today()

    # Initialize combined portfolio and individual stock contributions
    combined_portfolio = [0] * BACKTEST_PERIOD
    buy_and_hold_portfolio = [0] * BACKTEST_PERIOD
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
        trade_logs[ticker] = env.trade_log
        individual_portfolios[ticker] = pad_portfolio_history(env.portfolio_history, BACKTEST_PERIOD)

        # Analyze trades for the current stock
        analyze_trades_per_stock(env.trade_log, ticker, final_price=float(df["Close"].iloc[-1]))

        # Add the portfolio history to the combined portfolio
        for i in range(len(combined_portfolio)):
            combined_portfolio[i] += individual_portfolios[ticker][i]

        # Calculate buy-and-hold portfolio value for this stock
        initial_price = float(df["Close"].iloc[0])
        shares_held = balance_per_stock / initial_price
        stock_buy_and_hold_value = [float(shares_held * df["Close"].iloc[i]) for i in range(len(df))]
        stock_buy_and_hold_value = pad_portfolio_history(stock_buy_and_hold_value, BACKTEST_PERIOD)
        buy_and_hold_portfolio = [
            buy_and_hold_portfolio[i] + stock_buy_and_hold_value[i]
            for i in range(len(buy_and_hold_portfolio))
        ]

    # Render the combined portfolio results
    print("\n📊 Combined Portfolio Value Over Time with Individual Contributions")
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

    return combined_portfolio, buy_and_hold_portfolio

if __name__ == "__main__":
    combined_portfolio, buy_and_hold_portfolio = main()
    final_combined_value = combined_portfolio[-1]
    final_buy_and_hold_value = buy_and_hold_portfolio[-1]
    print(f"\n💰 Final Combined Portfolio Value: ${final_combined_value:.2f}")
    print(f"💰 Final Buy-and-Hold Portfolio Value: ${final_buy_and_hold_value:.2f}")
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(combined_portfolio, label="Combined Portfolio", linewidth=2, color="black")
    plt.plot(buy_and_hold_portfolio, label="Buy-and-Hold Portfolio", linestyle="--", color="blue")
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.savefig("plots/final_portfolio_comparison.png")
    plt.show()