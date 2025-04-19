import gymnasium as gym  # Use Gymnasium instead of OpenAI Gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

class TradingEnvRL(gym.Env):
    """Custom trading environment for reinforcement learning."""

    def __init__(self):
        super(TradingEnvRL, self).__init__()
        # Define the observation space (e.g., price, cash, shares)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        # Define the action space (e.g., buy, sell, hold)
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell

        # Initialize environment state
        self.current_step = 0
        self.cash = 10000  # Starting cash
        self.shares = 0  # Starting shares
        self.prices = np.random.rand(100) * 100  # Simulated price data
        self.done = False

    def reset(self, seed=None, **kwargs):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)  # Ensure compatibility with Gymnasium
        if seed is not None:
            np.random.seed(seed)  # Set the random seed for reproducibility
        self.current_step = 0
        self.cash = 10000
        self.shares = 0
        self.done = False
        # Return observation and an empty info dictionary
        return self._get_observation(), {}

    def step(self, action):
        """Execute one time step within the environment."""
        current_price = self.prices[self.current_step]

        # Execute action
        if action == 1:  # Buy
            max_shares = int(self.cash / current_price)
            self.shares += max_shares
            self.cash -= max_shares * current_price
        elif action == 2:  # Sell
            self.cash += self.shares * current_price
            self.shares = 0

        # Update to the next step
        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            self.done = True

        # Calculate reward (e.g., portfolio value change)
        portfolio_value = self.cash + self.shares * current_price
        reward = portfolio_value  # Reward is the portfolio value

        # Define termination and truncation flags
        terminated = self.done  # Episode ends when we reach the last step
        truncated = False  # No truncation logic in this environment

        return self._get_observation(), reward, terminated, truncated, {}

    def _get_observation(self):
        """Get the current observation."""
        current_price = self.prices[self.current_step]
        return np.array([current_price, self.cash, self.shares], dtype=np.float32)

# Train the RL agent
print("Starting PPO training...")
env = TradingEnvRL()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
print("PPO training completed.")

# Use the trained model for trading
print("Starting trading with the trained model...")
obs, _ = env.reset()  # Extract only the observation from the reset method
for _ in range(len(env.prices)):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {rewards}, Terminated: {terminated}, Truncated: {truncated}")
    if terminated or truncated:  # Check both termination and truncation flags
        break
print("Trading completed.")
