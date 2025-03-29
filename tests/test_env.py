import unittest
import numpy as np
import gym
from src.env.simple_trading_env import SimpleTradingEnv

class TestSimpleTradingEnv(unittest.TestCase):
    def setUp(self):
        prices = np.array([100, 101, 102, 103, 104])  # Sample price data
        self.env = SimpleTradingEnv(prices)

    def test_reset(self):
        obs = self.env.reset()
        self.assertEqual(self.env.step_idx, 0)
        self.assertEqual(self.env.cash, 10000)
        self.assertEqual(self.env.shares, 0)
        self.assertTrue(np.array_equal(obs, np.array([100, 10000, 0])))

    def test_buy_action(self):
        self.env.reset()
        action = 1  # Buy
        obs, reward, done, _ = self.env.step(action)
        self.assertEqual(self.env.shares, 1)
        self.assertEqual(self.env.cash, 9900)  # 10000 - 100

    def test_sell_action(self):
        self.env.reset()
        self.env.step(1)  # Buy
        action = 2  # Sell
        obs, reward, done, _ = self.env.step(action)
        self.assertEqual(self.env.shares, 0)
        self.assertEqual(self.env.cash, 10000)  # 9900 + 100

    def test_hold_action(self):
        self.env.reset()
        action = 0  # Hold
        obs, reward, done, _ = self.env.step(action)
        self.assertEqual(self.env.shares, 0)
        self.assertEqual(self.env.cash, 10000)

    def test_done_condition(self):
        self.env.reset()
        for _ in range(len(self.env.prices) - 1):
            self.env.step(0)  # Hold
        obs, reward, done, _ = self.env.step(0)  # Last step
        self.assertTrue(done)

if __name__ == '__main__':
    unittest.main()