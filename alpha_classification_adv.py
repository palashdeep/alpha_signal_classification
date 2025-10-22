# More Advanced Implementation: Alpha Signal Classification using XGBoost
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import xgboost as xgb

seed = 42

def get_data(tickers=["SPY"], start="2015-01-01", end=None):
    """Fetches and processes data from yfinance, returns feature DataFrame"""
    df = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        rows = []
        for t in tickers:
            if t not in tickers:
                continue
            dft = df[t].copy()
            dft['Ticker'] = t
            rows.append(dft)
        df = pd.concat(rows, axis=0)

    else:
        df = pd.DataFrame()
        for t in tickers:
            dft = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if dft.empty:
                continue
            dft = dft[['Open','High','Low','Close','Volume']].dropna().copy()
            dft['Ticker'] = t
            rows.append(dft)
            df = pd.concat(rows, axis=0)

    df = df.reset_index().rename(columns={'Date':'date'})
    df = df.sort_index()
    
    if df.empty:
        raise RuntimeError("No data downloaded. Check ticker or internet.")

    return df

def get_features_and_labels(group_df, horizon=1):
    """Generates features and labels for a given ticker group"""
    group = group_df.copy()
    group.sort_index(inplace=True)
    # returns
    group['ret_1'] = group['Close'].pct_change()                      # 1-day return
    group['ret_2'] = group['Close'].pct_change(2)
    group['ret_5'] = group['Close'].pct_change(5)

    # momentum / moving avgs
    group['ma5'] = group['Close'].rolling(5).mean()
    group['ma10'] = group['Close'].rolling(10).mean()
    group['ma_ratio'] = group['ma5'] / group['ma10'] - 1

    # volatility
    group['vol_5'] = group['ret_1'].rolling(5).std()
    group['vol_10'] = group['ret_1'].rolling(10).std()

    # volume features
    group['vol_zscore'] = (group['Volume'] - group['Volume'].rolling(20).mean()) / group['Volume'].rolling(20).std()

    # price position in BB-like band
    group['rolling_mean_20'] = group['Close'].rolling(20).mean()
    group['rolling_std_20'] = group['Close'].rolling(20).std()
    group['bb_pct'] = (group['Close'] - group['rolling_mean_20']) / (1e-9 + group['rolling_std_20'])

    group = group.dropna().copy()

    # Labels
    group['future_Close'] = group.groupby('Ticker')['Close'].shift(-horizon)
    group['future_ret'] = group['future_Close'] / group['Close'] - 1
    group['label'] = (group['future_ret'] > 0).astype(int)    # 1 = +ve return, 0 = -ve or zero return

    group = group.dropna().copy()
    
    return group

def train_test_split(df, features, test_size_frac=0.2):
    """Splits data into train and test sets based on time"""
    unique_dates = df['date'].sort_values().unique()
    split_idx = int((1 - test_size_frac) * len(unique_dates))
    split_date = unique_dates[split_idx]
    train_df = df[df['date'] < split_date].copy()
    test_df  = df[df['date'] >= split_date].copy()

    X_train = train_df[features].values
    y_train = train_df['label'].values
    X_test  = test_df[features].values
    y_test  = test_df['label'].values

    return X_train, y_train, X_test, y_test, train_df, test_df

def get_scaled_data(X_train, X_test):
    """Scales features using StandardScaler"""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def print_metrics(y_true, y_pred, probs=None):
    print("\n-----------Preformance Metrics------------")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
    try:
        print("AUC:", roc_auc_score(y_true, probs))
    except:
        pass
    print(classification_report(y_true, y_pred, digits=3))

def backtest_strategy(df_test):
    """Backtests a simple strategy based on predicted probabilities with equal-weighted returns"""
    returns = [] #dict of date, strategy return, number of stocks bought

    for date, group in df_test.groupby('date'):
        if group['label'].sum() == 0:
            returns.append({'date': date, 'strat_ret': 0.0, 'n_signal': 0})
            continue
        strat_ret = group.loc[group['pred_signal'] == 1, 'future_ret'].mean()
        n_signal = int(group['pred_signal'].sum())
        returns.append({'date': date, 'strat_ret': strat_ret, 'n_signal': n_signal})
    
    return pd.DataFrame(returns).sort_values('date').reset_index(drop=True)

df = get_data(tickers=["SPY", "AAPL", "MSFT", "GOOG"], start="2015-01-01", end=None)
df_features = df.groupby('Ticker').apply(get_features_and_labels)
features = ['ret_1','ret_2','ret_5','ma_ratio','vol_5','vol_10','vol_zscore','bb_pct']
df_features = df_features.dropna().copy()
df_features = df_features.reset_index(drop=True)

df_model = df_features[['date','Ticker'] + features + ['label', 'future_ret']].copy()
df_model = df_model.sort_values('date').reset_index(drop=True)

X_train, y_train, X_test, y_test, df_train, df_test = train_test_split(df_model, features, test_size_frac=0.2)
X_train_s, X_test_s, scaler = get_scaled_data(X_train, X_test)

# XGBoost Classifier with early stopping
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed)
xgb_clf.fit(X_train, y_train)
pred_prob = xgb_clf.predict_proba(X_test)[:,1]
y_pred = (pred_prob >= 0.5).astype(int)

print_metrics(y_test, y_pred, probs=pred_prob)

df_test = df_test.reset_index(drop=True)
df_test['pred_prob'] = pred_prob
df_test['pred_signal'] = y_pred  # 1 or 0

strategy_pref = backtest_strategy(df_test)
strategy_pref['total'] = (1 + strategy_pref['strat_ret']).cumprod()
# buy and hold benchmark
benchmark = df_test.groupby('date')['future_ret'].mean().reset_index().sort_values('date')
benchmark['total'] = (1 + benchmark['future_ret']).cumprod()

# merge into one
final_df = pd.merge(strategy_pref[['date','total']], benchmark[['date','total']], on='date', how='left', suffixes=('_strat','_bench'))

print("\n--------------Simple Strategy------------------")
print("Test period start:", strategy_pref['date'].min().date(), "end:", strategy_pref['date'].max().date())
print("Strategy total return (equal-weighted portfolio):", final_df['total_strat'].iloc[-1] - 1)
print("Benchmark return:", final_df['total_bench'].iloc[-1] - 1)
print("Avg # signals per day:", strategy_pref['n_signal'].mean())


plt.figure(figsize=(10,5))
plt.plot(final_df['date'], final_df['total_strat'], label='Strategy')
plt.plot(final_df['date'], final_df['total_bench'], label='Benchmark')
plt.title("Total returns")
plt.legend()
plt.tight_layout()
plt.show()