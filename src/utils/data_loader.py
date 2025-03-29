import yfinance as yf
import pandas as pd

def fetch_historical_data(ticker, period='1y'):
    data = yf.download(ticker, period=period, auto_adjust=True)
    return data

def preprocess_data(data):
    data = data[['Close']].copy()
    data['Returns'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    return data.to_numpy()