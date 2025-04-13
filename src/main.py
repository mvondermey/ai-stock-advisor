import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
from typing import List
from tqdm import tqdm
import time  # Import time module for delay

# Set a non-interactive backend to avoid RuntimeError
matplotlib.use("Agg")  # Use a non-interactive backend for rendering

# --- Constants ---
INITIAL_BALANCE = 20000  # Updated initial balance to 50000
TRANSACTION_COST = 0.0015  # Adjusted transaction cost
POSITION_SIZE = 1.0  # Use full capital for testing each trade
BACKTEST_PERIOD = 90  # Backtest period in terms of steps (e.g., trading days)
STOP_LOSS = 0.2  # 20% stop-loss
TAKE_PROFIT = 0.2  # 20% take-profit

# Define fixed stop-loss thresholds for specific stocks
FIXED_STOP_LOSS = {
}

# Define fixed take-profit thresholds for specific stocks
FIXED_TAKE_PROFIT = {
}

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

    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        self.cash = INITIAL_BALANCE
        self.shares = 0
        self.portfolio_history = [self.cash]
        self.trade_log = []

    def step(self):
        """Execute one step with buy-on-uptrend and sell-on-downtrend logic."""
        # Ensure scalar values for current_price and previous_price
        current_price = self.df["Close"].iloc[self.current_step]
        if isinstance(current_price, pd.Series):
            current_price = current_price.item()

        previous_price = (
            self.df["Close"].iloc[self.current_step - 1]
            if self.current_step > 0
            else current_price
        )
        if isinstance(previous_price, pd.Series):
            previous_price = previous_price.item()

        # Debug: Print current step, price, cash, and shares
        #print(f"Step {self.current_step}: Current Price: {current_price:.2f}, Cash: {self.cash:.2f}, Shares: {self.shares}")

        # Buy logic: Buy if price is increasing and no shares are held
        if self.shares == 0 and current_price > previous_price:
            max_shares = int((self.cash * POSITION_SIZE) / current_price)  # Calculate max shares to buy
            if max_shares > 0:
                self.shares = max_shares
                self.cash -= max_shares * current_price * (1 + self.transaction_cost)  # Deduct cash for the purchase
                self.trade_log.append((self.current_step, "BUY", current_price, self.shares))  # Log the trade
                print(f"Step {self.current_step}: Bought {max_shares} shares at {current_price:.2f}")

        # Sell logic: Sell if price is decreasing and shares are held
        elif self.shares > 0 and current_price < previous_price:
            self.cash += self.shares * current_price * (1 - self.transaction_cost)  # Add cash from the sale
            self.trade_log.append((self.current_step, "SELL", current_price, self.shares))  # Log the trade
            print(f"Step {self.current_step}: Sold {self.shares} shares at {current_price:.2f}")
            self.shares = 0  # Reset shares to 0

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
        plt.savefig(f"portfolio_{id(self)}.png")  # Save the plot to a file
        plt.show()  # Display the plot
        plt.close()  # Explicitly close the plot to avoid conflicts

# --- Data fetching and preprocessing ---
def prepare_data(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download and preprocess data for a given ticker and date range."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True).dropna()
    if df.empty or len(df) < 20:  # Require at least 20 rows for indicators
        raise ValueError(f"Insufficient data for {ticker} from {start} to {end}")
    return df

def calculate_volatility(df: pd.DataFrame) -> float:
    """Calculate historical volatility for a stock."""
    returns = df["Close"].pct_change().dropna()
    return returns.std().item()  # Ensure the result is a scalar

def get_top_performing_stocks_ytd(sp500: bool = True, n: int = 10) -> List[str]:
    """Get top n performing stocks YTD from S&P 500."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    symbol_column = "Symbol"  # Column name for S&P 500
    index_data = pd.read_html(url)[0]
    if symbol_column not in index_data.columns:
        raise KeyError(f"Column '{symbol_column}' not found in the table fetched from {url}")

    tickers = [symbol.replace('.', '-') for symbol in index_data[symbol_column].tolist()]

    performances = []
    end_date = datetime.today()
    start_date = datetime(end_date.year, 1, 1)  # Start of the current year
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
    """Analyze trade performance."""
    buys = [trade for trade in trade_log if trade[1] == "BUY"]
    sells = [trade for trade in trade_log if trade[1] == "SELL"]
    profits = [sells[i][2] - buys[i][2] for i in range(min(len(buys), len(sells)))]
    total_profit = sum(profits)
    win_rate = len([p for p in profits if p > 0]) / len(profits) if profits else 0
    avg_profit = total_profit / len(profits) if profits else 0
    print(f"Total Profit: ${total_profit:.2f}, Win Rate: {win_rate:.2%}, Avg Profit: ${avg_profit:.2f}")

def analyze_trades_per_stock(trade_log: List[tuple], ticker: str):
    """Analyze trade performance for a specific stock."""
    buys = [trade for trade in trade_log if trade[1] == "BUY"]
    sells = [trade for trade in trade_log if trade[1] == "SELL"]
    profits = [sells[i][2] - buys[i][2] for i in range(min(len(buys), len(sells)))]
    total_profit = sum(profits)
    win_rate = len([p for p in profits if p > 0]) / len(profits) if profits else 0
    losses = len([p for p in profits if p <= 0])
    print(f"\n📊 {ticker} Trade Analysis:")
    print(f"  Total Profit: ${total_profit:.2f}")
    print(f"  Wins: {len([p for p in profits if p > 0])}, Losses: {losses}")
    print(f"  Win Rate: {win_rate:.2%}")

def calculate_weights(trade_logs: dict) -> dict:
    """Calculate weights for each stock based on their performance."""
    weights = {}
    total_profit = 0

    # Calculate total profit across all stocks
    for ticker, log in trade_logs.items():
        buys = [trade for trade in log if trade[1] == "BUY"]
        sells = [trade for trade in log if trade[1] == "SELL"]
        profits = [sells[i][2] - buys[i][2] for i in range(min(len(buys), len(sells)))]
        stock_profit = sum(profits)
        weights[ticker] = max(stock_profit, 0)  # Only consider positive profits
        total_profit += max(stock_profit, 0)

    # Normalize weights
    for ticker in weights:
        weights[ticker] = weights[ticker] / total_profit if total_profit > 0 else 1 / len(trade_logs)

    return weights

def pad_portfolio_history(portfolio_history: List[float], max_steps: int) -> List[float]:
    """Pad the portfolio history to ensure it matches the maximum number of steps."""
    if len(portfolio_history) < max_steps:
        last_value = portfolio_history[-1] if portfolio_history else 0
        portfolio_history.extend([last_value] * (max_steps - len(portfolio_history)))
    return portfolio_history

# --- Main function ---
def main():
    """Main execution function."""
    # Ensure the 'plots' directory exists
    os.makedirs("plots", exist_ok=True)

    print("\n" + "="*50)
    print("🚀 Rule-Based Trading System")
    print("="*50 + "\n")

    print("🔍 Fetching top-performing stocks from S&P 500...")
    top_tickers = get_top_performing_stocks_ytd(sp500=True, n=5)  # Fetch top 5 performing stocks YTD
    print(f"📈 Selected tickers: {', '.join(top_tickers)}\n")

    # Define the backtest period
    end_date = datetime.today()
    start_date = end_date - timedelta(days=BACKTEST_PERIOD)

    # Initialize combined portfolio value and individual stock contributions
    combined_portfolio = [0] * BACKTEST_PERIOD  # Start with 0 for combined portfolio
    buy_and_hold_portfolio = [0] * BACKTEST_PERIOD  # Start with 0 for buy-and-hold portfolio
    individual_portfolios = {}  # Store individual stock portfolio histories
    trade_logs = {}  # Store trade logs for each stock
    stop_loss_thresholds = {}  # Store dynamic stop-loss thresholds for each stock

    # Allocate the initial balance equally across all selected stocks
    balance_per_stock = INITIAL_BALANCE / len(top_tickers)

    for ticker in top_tickers:
        print(f"\n📈 Fetching data for {ticker}...")
        try:
            df = prepare_data(ticker, start_date, end_date)

            # Dynamically adjust stop-loss thresholds
            volatility = calculate_volatility(df)
            stop_loss_threshold = min(max(volatility * 2, 0.05), 0.5)  # Clamp between 5% and 50%
            stop_loss_thresholds[ticker] = stop_loss_threshold
            print(f"🔍 Adjusted stop-loss for {ticker}: {stop_loss_threshold:.2%}")

            print(f"🔄 Running backtest for {ticker}...")
            env = TradingEnv(df, initial_balance=balance_per_stock)  # Use allocated balance per stock
            env.stop_loss = stop_loss_threshold  # Set dynamic stop-loss
            env.run()

            # Store the trade log and portfolio history for weight calculation and plotting
            trade_logs[ticker] = env.trade_log
            individual_portfolios[ticker] = pad_portfolio_history(env.portfolio_history, BACKTEST_PERIOD)

            # Debug: Print trade log for the current stock
            print(f"🔍 Trade log for {ticker}: {env.trade_log}")

            # Render and save the individual stock portfolio plot
            print(f"📊 Rendering portfolio plot for {ticker}...")
            plt.figure(figsize=(10, 5))
            plt.plot(individual_portfolios[ticker], label=f"{ticker} Portfolio")
            plt.title(f"Portfolio Value Over Time for {ticker}")
            plt.xlabel("Steps")
            plt.ylabel("Portfolio Value ($)")
            plt.legend()
            plt.savefig(f"plots/portfolio_{ticker}.png")  # Save the plot for the stock in the 'plots' directory
            plt.close()  # Explicitly close the plot to avoid conflicts

            # Analyze trades for the current stock
            analyze_trades_per_stock(env.trade_log, ticker)

            # Add the portfolio history to the combined portfolio
            for i in range(len(combined_portfolio)):
                combined_portfolio[i] += individual_portfolios[ticker][i]

            # Calculate buy-and-hold portfolio value for this stock
            initial_price = float(df["Close"].iloc[0])  # Ensure scalar value
            shares_held = balance_per_stock / initial_price
            stock_buy_and_hold_value = [float(shares_held * df["Close"].iloc[i]) for i in range(len(df))]

            # Pad the buy-and-hold portfolio to match BACKTEST_PERIOD
            stock_buy_and_hold_value = pad_portfolio_history(stock_buy_and_hold_value, BACKTEST_PERIOD)

            # Add this stock's buy-and-hold value to the combined buy-and-hold portfolio
            buy_and_hold_portfolio = [
                buy_and_hold_portfolio[i] + stock_buy_and_hold_value[i]
                for i in range(len(buy_and_hold_portfolio))
            ]

        except Exception as e:
            print(f"❌ Failed to process {ticker}: {e}")

    # Ensure combined_portfolio and buy_and_hold_portfolio are lists of scalars
    combined_portfolio = [float(value) for value in combined_portfolio]
    buy_and_hold_portfolio = [float(value) for value in buy_and_hold_portfolio]

    # Render the combined portfolio results with individual stock contributions
    print("\n📊 Combined Portfolio Value Over Time with Individual Contributions")
    plt.figure(figsize=(12, 6))
    for ticker, portfolio in individual_portfolios.items():
        plt.plot(portfolio, label=f"{ticker} Portfolio")
    plt.plot(combined_portfolio, label="Combined Portfolio", linewidth=2, color="black")
    plt.plot(buy_and_hold_portfolio, label="Buy-and-Hold Portfolio", linestyle="--", color="blue")
    plt.title("Combined Portfolio Value Over Time with Individual Contributions")
    plt.xlabel("Steps")
    plt.ylabel("Portfolio Value ($)")
    plt.legend(title="Portfolio Type")
    plt.ylim(0, max(max(combined_portfolio), max(buy_and_hold_portfolio)) * 1.1)  # Adjust y-axis to fit the max value
    plt.savefig("plots/combined_portfolio_with_individuals.png")  # Save the combined portfolio plot in the 'plots' directory
    plt.close()  # Explicitly close the plot to avoid conflicts

    # Add a delay to keep the plots open
    print("⏳ Keeping plots open for 10 seconds...")
    time.sleep(10)  # Delay for 10 seconds

if __name__ == "__main__":
    main()