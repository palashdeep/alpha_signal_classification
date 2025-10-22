"""
===================================
Mean Reversion Strategy and Backtesting Framework
===================================
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import os

def fetch_data(tickers, start=None, end=None, period='10y', interval='1d'):
    """Fetch OHLCV using yfinance. Returns a dict of DataFrames (close-adj included).
    """
    data = {}
    for t in tickers:
        df = yf.download(t, start=start, end=end, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {t}. Check ticker or network.")
        df = df.dropna()
        df['close'] = df['Close']
        data[t] = df
    return data

def single_asset_mean_reversion(close, ma_window=20, std_window=20, entry_z=2.0, exit_z=0.0,
                                pct_per_trade=0.1, tc=0.0005, slippage=0.0002):
    """Backtest simple Z-score mean reversion on a single close price series.

    Parameters
    ----------
    close : pd.Series
    ma_window, std_window : int
    entry_z, exit_z : float
    pct_per_trade : fraction of equity to allocate per trade (position sizing)
    tc : round-trip transaction cost as fraction of trade value
    slippage : per-trade slippage fraction

    Returns
    -------
    results : dict with trades, equity curve, and metrics
    """
    prices = close.copy().ffill().dropna()
    ma = prices.rolling(ma_window).mean()
    sigma = prices.rolling(std_window).std()
    z = (prices - ma) / sigma

    # state: 0 flat, 1 long, -1 short
    state = 0
    entry_price = 0.0
    equity = 1.0
    equity_curve = []
    positions = []
    trade_logs = []

    for t in range(len(prices)):
        date = prices.index[t]
        p = prices.iat[t]
        zt = z.iat[t]

        # entry
        if state == 0:
            if zt <= -entry_z:
                # go long
                state = 1
                entry_price = p * (1 + slippage)
                position_value = equity * pct_per_trade
                shares = position_value / entry_price
                cost = position_value * tc
                equity -= cost
                trade_logs.append({'entry_date': date, 'side': 'long', 'entry_price': entry_price, 'shares': shares})
            elif zt >= entry_z:
                # go short
                state = -1
                entry_price = p * (1 - slippage)
                position_value = equity * pct_per_trade
                shares = position_value / entry_price
                cost = position_value * tc
                equity -= cost
                trade_logs.append({'entry_date': date, 'side': 'short', 'entry_price': entry_price, 'shares': shares})

        # exit
        elif state == 1:
            if zt >= exit_z:
                exit_price = p * (1 - slippage)
                last = trade_logs[-1]
                pnl = last['shares'] * (exit_price - last['entry_price'])
                equity += pnl
                cost = (last['shares'] * exit_price) * tc
                equity -= cost
                last.update({'exit_date': date, 'exit_price': exit_price, 'pnl': pnl})
                state = 0
        elif state == -1:
            if zt <= exit_z:
                exit_price = p * (1 + slippage)
                last = trade_logs[-1]
                pnl = last['shares'] * (last['entry_price'] - exit_price)
                equity += pnl
                cost = (last['shares'] * exit_price) * tc
                equity -= cost
                last.update({'exit_date': date, 'exit_price': exit_price, 'pnl': pnl})
                state = 0

        equity_curve.append({'date': date, 'equity': equity, 'price': p, 'z': zt, 'state': state})

    ec = pd.DataFrame(equity_curve).set_index('date')

    # metrics
    returns = ec['equity'].pct_change().fillna(0)
    total_return = ec['equity'].iloc[-1] / ec['equity'].iloc[0] - 1
    ann_factor = 252 * 1  # daily data assumption
    if len(returns) > 1:
        daily_ret = returns
        sr = (daily_ret.mean() / daily_ret.std()) * np.sqrt(ann_factor) if daily_ret.std() != 0 else np.nan
    else:
        sr = np.nan
    maxdd = (ec['equity'].cummax() - ec['equity']).max()

    results = {
        'equity_curve': ec,
        'trades': pd.DataFrame(trade_logs),
        'metrics': {'total_return': total_return, 'sharpe': sr, 'max_drawdown': maxdd}
    }
    return results


# ------------------------- Pairs Mean Reversion -------------------------

def pairs_mean_reversion(prices_x, prices_y, lookback=60, z_entry=2.0, z_exit=0.5,
                          pct_per_trade=0.1, tc=0.0005, slippage=0.0002):
    """Simple pairs trading using OLS residuals as spread.

    We run a rolling OLS over the lookback window to compute residuals, z-score
    the residuals, and trade when spread z exceeds entry threshold.
    """
    # align
    df = pd.DataFrame({'x': prices_x, 'y': prices_y}).dropna()
    dates = df.index

    equity = 1.0
    equity_curve = []
    trades = []
    state = 0

    for i in range(lookback, len(df)):
        window = df.iloc[i - lookback:i]
        y = window['y'].values
        X = add_constant(window['x'].values)
        model = OLS(y, X).fit()
        beta = model.params[1]
        const = model.params[0]

        # current spread
        cur_x = df['x'].iat[i]
        cur_y = df['y'].iat[i]
        spread = cur_y - (const + beta * cur_x)

        # compute z based on residuals in window
        resids = model.resid
        z = (spread - resids.mean()) / resids.std()
        date = dates[i]

        # entry
        if state == 0:
            if z >= z_entry:
                # short y, long x
                # value allocated
                pos_value = equity * pct_per_trade
                # number of units chosen such that dollar exposure roughly matches
                # for simplicity, long shares_x = pos_value / x, short shares_y = pos_value / y
                shares_x = pos_value / cur_x
                shares_y = pos_value / cur_y
                entry = {'entry_date': date, 'side': 'short_spread', 'beta': beta, 'const': const,
                         'shares_x': shares_x, 'shares_y': shares_y, 'entry_spread': spread}
                trades.append(entry)
                cost = pos_value * tc
                equity -= cost
                state = 1
            elif z <= -z_entry:
                # long y, short x
                pos_value = equity * pct_per_trade
                shares_x = pos_value / cur_x
                shares_y = pos_value / cur_y
                entry = {'entry_date': date, 'side': 'long_spread', 'beta': beta, 'const': const,
                         'shares_x': shares_x, 'shares_y': shares_y, 'entry_spread': spread}
                trades.append(entry)
                cost = pos_value * tc
                equity -= cost
                state = -1

        # exit
        elif state == 1:
            # short spread, wait until spread mean reversion
            if z <= z_exit:
                last = trades[-1]
                exit_price_x = cur_x * (1 - slippage)
                exit_price_y = cur_y * (1 + slippage)
                pnl = last['shares_x'] * (exit_price_x - last.get('entry_x_price', last['shares_x'] * cur_x))
                # For simplicity compute spread pnl more directly: change in (y - beta*x)
                pnl = (last['shares_y'] * (last['entry_spread'] - spread))
                equity += pnl
                cost = (last['shares_x'] * exit_price_x + last['shares_y'] * exit_price_y) * tc
                equity -= cost
                last.update({'exit_date': date, 'exit_spread': spread, 'pnl': pnl})
                state = 0
        elif state == -1:
            if z >= -z_exit:
                last = trades[-1]
                exit_price_x = cur_x * (1 + slippage)
                exit_price_y = cur_y * (1 - slippage)
                pnl = (last['shares_y'] * (spread - last['entry_spread']))
                equity += pnl
                cost = (last['shares_x'] * exit_price_x + last['shares_y'] * exit_price_y) * tc
                equity -= cost
                last.update({'exit_date': date, 'exit_spread': spread, 'pnl': pnl})
                state = 0

        equity_curve.append({'date': date, 'equity': equity, 'spread': spread, 'z': z, 'state': state})

    ec = pd.DataFrame(equity_curve).set_index('date')
    trades_df = pd.DataFrame(trades)

    # metrics
    returns = ec['equity'].pct_change().fillna(0)
    total_return = ec['equity'].iloc[-1] / ec['equity'].iloc[0] - 1 if len(ec) else 0.0
    if len(returns) > 1:
        sr = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else np.nan
    else:
        sr = np.nan
    maxdd = (ec['equity'].cummax() - ec['equity']).max() if not ec.empty else np.nan

    return {'equity_curve': ec, 'trades': trades_df, 'metrics': {'total_return': total_return, 'sharpe': sr, 'max_drawdown': maxdd}}


# ------------------------------ Plot helpers ------------------------------

def plot_equity(ec, title, fname=None):
    """Equity evolution plot"""
    plt.figure(figsize=(10, 5))
    plt.plot(ec.index, ec['equity'])
    plt.title(title)
    plt.grid(True)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_price_with_z(prices, z, title, fname=None):
    """How often we trade"""
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    ax[0].plot(prices.index, prices, label='price')
    ax[0].set_title(title)
    ax[1].plot(z.index, z, label='z')
    ax[1].axhline(0, color='black', linewidth=0.7)
    ax[1].axhline(2, color='grey', linestyle='--', linewidth=0.7)
    ax[1].axhline(-2, color='grey', linestyle='--', linewidth=0.7)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_pnl_distribution(trades_df, title, fname=None):
    """PnL distribution histogram"""
    plt.figure(figsize=(8, 4))
    plt.hist(trades_df['pnl'], bins=30, alpha=0.7)
    plt.title(title)
    plt.xlabel("P&L")
    plt.ylabel("Frequency")
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def plot_drawdowns(ec, title, fname=None):
    """Drawdown plot"""
    cummax = ec['equity'].cummax()
    drawdown = ec['equity'] - cummax

    plt.figure(figsize=(10, 4))
    plt.plot(ec.index, drawdown, color='red')
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.grid(True)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def performance_scatter(trades_df, title, fname=None):
    """Performance scatter plot"""
    trades_df['duration'] = (pd.to_datetime(trades_df['exit_date']) - pd.to_datetime(trades_df['entry_date'])).dt.days
    plt.figure(figsize=(8, 5))
    plt.scatter(trades_df['duration'], trades_df['pnl'], alpha=0.6)
    plt.axhline(0, color='black', linestyle='--')
    plt.title(title)
    plt.xlabel("Duration (days)")
    plt.ylabel("P&L")
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def price_with_trade_markers(df, trades_df, title, fname=None):
    """Price with trade markers"""
    plt.figure(figsize=(12, 6))
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

    plt.title(title)
    plt.legend()
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
