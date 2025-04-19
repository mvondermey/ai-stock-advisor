from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

def train_predictive_model(df: pd.DataFrame):
    """
    Train a predictive model using the provided DataFrame.
    :param df: DataFrame containing features and target variable.
    :return: Trained model.
    """
    print(f"Initial dataset size: {len(df)} rows")
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
    df['Returns'] = df['Close'].pct_change()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()

    latest_data = df[['Returns', 'SMA_10', 'SMA_30', 'Volatility']].iloc[-1].values.reshape(1, -1)
    return model.predict(latest_data)[0]
