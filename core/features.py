import pandas as pd

def compute_features(df):
    """Computes features for a DataFrame"""

    df = df.copy()

    df['ret_1'] = df['Close'].pct_change()
    df['ret_2'] = df['Close'].pct_change(2)
    df['ret_5'] = df['Close'].pct_change(5)

    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma10'] = df['Close'].rolling(10).mean()
    df['ma_ratio'] = df['ma5'] / df['ma10'] - 1

    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['vol_10'] = df['ret_1'].rolling(10).std()

    df['vol_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

    df['bb_pct'] = (df['Close'] - df['Close'].rolling(20).mean()) / (1e-9 + df['Close'].rolling(20).std())

    return df.dropna()

def future_return_labels(df, horizon=1):
    """Generates future return labels by shifting with horizon"""
    
    df = df.copy()
    df['future_close'] = df['Close'].shift(-horizon)
    df['future_ret'] = df['future_close'] / df['Close'] - 1
    df['label'] = (df['future_ret'] > 0).astype(int)

    return df.dropna()