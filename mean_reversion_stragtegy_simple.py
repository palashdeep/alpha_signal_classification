import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# PARAMETERS
TICKER = "QQQ"             # S&P 500 ETF
START = "2015-01-01"
END = None                 # None => today
HORIZON = 1                # trading days
TEST_SIZE_FRAC = 0.2
RANDOM_SEED = 42
LOOKBACK = 40              # days
THRESHOLD = 2.5            # z-score threshold for entry/exit

# Fetch data from yfinance
df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=True)
if df.empty:
    raise RuntimeError("No data downloaded. Check ticker or internet.")

df = df[['Open','High','Low','Close','Volume']].dropna()

if getattr(df.columns, 'nlevels', 1) > 1:
    # flatten MultiIndex columns to "Field_TICKER" strings
    df.columns = ['{}'.format(col[0]) for col in df.columns]
# print(df.columns)
df['ma'] = df['Close'].rolling(window=LOOKBACK).mean()

df['vol'] = df['Close'].rolling(window=LOOKBACK).std()

df['Z-score'] = (df['Close'] - df['ma']) / (1e-9 + df['vol'])

df = df.dropna().copy()

df['signal'] = 0
df.loc[df['Z-score'] <= -THRESHOLD, 'signal'] = 1
df.loc[df['Z-score'] >= THRESHOLD,  'signal'] = -1

dates = df.index.values
prices = df['Close'].values
signal = df['signal'].values
z_score = df['Z-score'].values

position = 0            # 1 = long, -1 = short, 0 = flat
entry_price = None
equity = 1000.0

pos_list = [0]
equity_list = [1000.0]
trade_logs = []


for i in range(1, len(df)):

    if position == 0:
        if signal[i] == 1:
            position = 1
            entry_price = prices[i]
            shares = equity // entry_price
            trade_logs.append({'entry_date': dates[i], 'side': 'long', 'entry_price': entry_price, 'shares': shares})
        elif signal[i] == -1:
            position = -1
            entry_price = prices[i]
            shares = equity // entry_price
            trade_logs.append({'entry_date': dates[i], 'side': 'short', 'entry_price': entry_price, 'shares': shares})
    
    elif position == 1:
        if signal[i] == -1 or z_score[i] >= 0:
            position = 0
            exit_price = prices[i]
            last = trade_logs[-1]
            pnl = (exit_price - last['entry_price']) * shares
            equity += pnl
            last.update({'exit_date': dates[i], 'exit_price': exit_price, 'pnl': pnl})
            shares = 0
            entry_price = None

    elif position == -1:
        if signal[i] == 1 or z_score[i] <= 0:
            position = 0
            exit_price = prices[i]
            last = trade_logs[-1]
            pnl = (last['entry_price'] - exit_price) * shares
            equity += pnl
            last.update({'exit_date': dates[i], 'exit_price': exit_price, 'pnl': pnl})
            shares = 0
            entry_price = None

    pos_list.append(position)
    equity_list.append(equity)

df['position'] = pos_list
df['equity'] = equity_list

trades_df = pd.DataFrame(trade_logs)

rets = df['equity'].pct_change().fillna(0)
sharpe = rets.mean() / rets.std() * np.sqrt(252)
win_rate = (trades_df['pnl'] > 0).mean()
avg_pnl = trades_df['pnl'].mean()

#Equity evolution plot
plt.figure(figsize=(10,5))
plt.plot(df.index, df['equity'], label='Equity Curve')
plt.title("Mean Reversion Strategy Equity Curve")
plt.ylabel("Equity")
plt.grid(True)
plt.legend()
plt.show()

# PnL
plt.figure(figsize=(8,4))
plt.hist(trades_df['pnl'], bins=30, alpha=0.7)
plt.title("Distribution of Trade P&L")
plt.xlabel("P&L")
plt.ylabel("Frequency")
plt.show()

#Drawdown
equity = df['equity']
cummax = equity.cummax()
drawdown = equity - cummax

plt.figure(figsize=(10,4))
plt.plot(df.index, drawdown, color='red')
plt.title("Drawdowns")
plt.ylabel("Drawdown")
plt.grid(True)
plt.show()

# Performance
trades_df['duration'] = (pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])).dt.days
plt.figure(figsize=(8,5))
plt.scatter(trades_df['duration'], trades_df['pnl'], alpha=0.6)
plt.axhline(0, color='black', linestyle='--')
plt.title("Trade Duration vs P&L")
plt.xlabel("Duration (days)")
plt.ylabel("P&L")
plt.show()

# How often we trade
fig, ax = plt.subplots(2, 1, figsize=(12,6), sharex=True)
ax[0].plot(df.index, df['Close'], label='Price')
ax[0].set_title("Price vs Z-score Signals")

ax[1].plot(df.index, df['Z-score'], label='Z-score')
ax[1].axhline(2, color='grey', linestyle='--')
ax[1].axhline(-2, color='grey', linestyle='--')
ax[1].axhline(0, color='black', linewidth=0.7)
ax[1].fill_between(df.index, -3, 3, where=(df['position']!=0), color='yellow', alpha=0.2, label='In Position')

ax[1].legend()
plt.show()

# Price with trade markers
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Price', alpha=0.6)

# plot trades
for _, row in trades_df.iterrows():
    if row['side'] == 'long':
        plt.scatter(row['entry_date'], row['entry_price'], marker='^', color='g', s=100, label='Long Entry' if 'Long Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
        if 'exit_date' in row and pd.notna(row['exit_date']):
            plt.scatter(row['exit_date'], row['exit_price'], marker='v', color='g', s=100, label='Long Exit' if 'Long Exit' not in plt.gca().get_legend_handles_labels()[1] else "")
    elif row['side'] == 'short':
        plt.scatter(row['entry_date'], row['entry_price'], marker='v', color='r', s=100, label='Short Entry' if 'Short Entry' not in plt.gca().get_legend_handles_labels()[1] else "")
        if 'exit_date' in row and pd.notna(row['exit_date']):
            plt.scatter(row['exit_date'], row['exit_price'], marker='^', color='r', s=100, label='Short Exit' if 'Short Exit' not in plt.gca().get_legend_handles_labels()[1] else "")

plt.title("Price with Trade Markers")
plt.legend()
plt.show()