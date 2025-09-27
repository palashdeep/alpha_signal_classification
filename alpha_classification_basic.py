# Basic Implementation: Alpha Signal Classification using Logistic Regression
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


# PARAMETERS
TICKER = "SPY"             # S&P 500 ETF
START = "2015-01-01"
END = None                 # None => today
HORIZON = 1                # trading days
TEST_SIZE_FRAC = 0.2
RANDOM_SEED = 42


# Fetch data from yfinance
df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=True)
if df.empty:
    raise RuntimeError("No data downloaded. Check ticker or internet.")

df = df[['Open','High','Low','Close','Volume']].dropna()

# returns
df['ret_1'] = df['Close'].pct_change()                      # 1-day return
df['ret_2'] = df['Close'].pct_change(2)
df['ret_5'] = df['Close'].pct_change(5)

# momentum / moving avgs
df['ma5'] = df['Close'].rolling(5).mean()
df['ma10'] = df['Close'].rolling(10).mean()
df['ma_ratio'] = df['ma5'] / df['ma10'] - 1

# volatility
df['vol_5'] = df['ret_1'].rolling(5).std()
df['vol_10'] = df['ret_1'].rolling(10).std()

# volume features
df['vol_zscore'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

# price position in BB-like band
df['rolling_mean_20'] = df['Close'].rolling(20).mean()
df['rolling_std_20'] = df['Close'].rolling(20).std()
df['bb_pct'] = (df['Close'] - df['rolling_mean_20']) / (1e-9 + df['rolling_std_20'])

df = df.dropna().copy()

# Labels
df['future_Close'] = df['Close'].shift(-HORIZON)
df['future_ret'] = df['future_Close'] / df['Close'] - 1
df['label'] = (df['future_ret'] > 0).astype(int)    # 1 = up, 0 = down or zero

df = df.dropna().copy()

features = ['ret_1','ret_2','ret_5','ma_ratio','vol_5','vol_10','vol_zscore','bb_pct']
X = df[features].values
y = df['label'].values

# Train-test split
n = len(df)
split = int((1 - TEST_SIZE_FRAC) * n)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
df_train, df_test = df.iloc[:split].copy(), df.iloc[split:].copy()

# scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(random_state=RANDOM_SEED, solver='lbfgs', max_iter=1000)
clf.fit(X_train_s, y_train)
probs_test = clf.predict_proba(X_test_s)[:,1]
preds_test = (probs_test >= 0.5).astype(int) # for prob >= 0.5, we predict up (1)

print("\n-------------Classification Metrics-------------\n")
print("Accuracy:", accuracy_score(y_test, preds_test))
print("Precision:", precision_score(y_test, preds_test))
print("Recall:", recall_score(y_test, preds_test))
print("F1:", f1_score(y_test, preds_test))
try:
    print("AUC:", roc_auc_score(y_test, probs_test))
except:
    pass
print("\nFull classification report:")
print(classification_report(y_test, preds_test, digits=3))

# Backtest using predicted direction
# go long if prob > threshold, else do nothing (no shorts for now)
threshold = 0.5
df_test = df_test.copy()
df_test['pred_prob'] = probs_test
df_test['pred_signal'] = (df_test['pred_prob'] >= threshold).astype(int)  # 1 or 0

df_test['strategy_ret'] = df_test['pred_signal'] * df_test['future_ret']
df_test['buy_and_hold_ret'] = df_test['future_ret']  # benchmark return without any strategy

# total returns
df_test['strategy_total'] = (1 + df_test['strategy_ret']).cumprod()
df_test['bench_total'] = (1 + df_test['buy_and_hold_ret']).cumprod()

print("\n-------------Simple Strategy-------------")
print("Period start:", df_test.index[0].date(), "end:", df_test.index[-1].date())
print("Strategy return:", df_test['strategy_total'].iloc[-1] - 1)
print("Buy & Hold (benchmark) return:", df_test['bench_total'].iloc[-1] - 1)
print("Number of trades:", int(df_test['pred_signal'].sum()))
print("Avg return:", df_test.loc[df_test['pred_signal']==1, 'future_ret'].mean())

# Plots
plt.figure(figsize=(10,5))
plt.plot(df_test.index, df_test['strategy_total'], label='Strategy')
plt.plot(df_test.index, df_test['bench_total'], label='Benchmark')
plt.title(f"Total returns on test period ({TICKER})")
plt.legend()
plt.tight_layout()
plt.show()

# feature importance approximation for logistic (coeff * std)
coefs = clf.coef_.flatten()
feat_imp = pd.Series(coefs * scaler.scale_, index=features).abs().sort_values(ascending=False)
print("\nApprox feature importance (|coef * scale|):")
print(feat_imp)