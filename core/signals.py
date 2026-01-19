import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def z_score_signal(price, lookback):
    """Calculate z-score of the price over a lookback period"""

    if len(price) < lookback or lookback < 1:
        raise ValueError("Lookback period must be less than or equal to the length of the price series and greater than 0.")
    
    ma = price.rolling(window=lookback).mean()
    vol = price.rolling(window=lookback).std()
    zscore = (price - ma) / (vol + 1e-9)

    return zscore

def fit_logistic(X_train, y_train):
    """Fit LR model"""

    scalar = StandardScaler()
    Xs = scalar.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(Xs, y_train)

    return model, scalar