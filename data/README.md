# Data Documentation for Trading AI Project

This directory contains data-related resources for the Trading AI project. 

## Data Sources
- Historical stock prices are fetched using the `yfinance` library, which provides access to financial data from Yahoo Finance.

## Data Format
- The primary data used in this project consists of stock price data, specifically the closing prices of stocks. The data is typically structured in a time series format.

## Usage
- The data is utilized within the custom Gym environment defined in `src/env/simple_trading_env.py` to simulate trading actions and evaluate the performance of the trading AI.

## Additional Notes
- Ensure that the necessary libraries are installed as specified in the `requirements.txt` file to facilitate data fetching and processing.