from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_predictive_model(df):
    """Train a predictive model using historical data."""
    df['Returns'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df = df.dropna()

    X = df[['Returns', 'SMA_10', 'SMA_30']].values
    y = df['Close'].shift(-1).dropna().values  # Predict next day's price

    model = RandomForestRegressor()
    model.fit(X[:-1], y)  # Train on all but the last row
    return model

def predict_next_price(model, df):
    """Predict the next day's price."""
    df['Returns'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df = df.dropna()

    latest_data = df[['Returns', 'SMA_10', 'SMA_30']].iloc[-1].values.reshape(1, -1)
    return model.predict(latest_data)[0]
