import numpy as np
import pandas as pd

def mean_reversion_backtest(price, z_score, entry_z, exit_z, initial_equity = 1000, cost_per_trade=0.0):
    """Backtest a simple mean reversion strategy based on z-score signals"""

    position = 0
    entry_price = None

    equity = initial_equity
    equity_curve = []
    trade_logs = []
    position_log = []

    for t in range(len(price)):
        p = price.iloc[t].item()
        zt = z_score.iloc[t].item()
        date = price.index[t]

        if position == 0:
            if zt <= -entry_z: # BUY
                position = 1
                entry_price = p
                trade_logs.append({'entry_date': date, 'side': 'long', 'entry_price': entry_price})
                equity -= cost_per_trade
            elif zt >= entry_z: # SELL
                position = -1
                entry_price = p
                trade_logs.append({'entry_date': date, 'side': 'short', 'entry_price': entry_price})
                equity -= cost_per_trade

        elif position == 1 and zt >= -exit_z:   # CLOSE LONG
            pnl = p - entry_price
            equity += pnl
            trade_logs[-1].update({'exit_date': date, 'exit_price': p, 'pnl': pnl})
            position = 0
            entry_price = None
            equity -= cost_per_trade
        
        elif position == -1 and zt <= exit_z:   # CLOSE SHORT
            pnl = entry_price - p
            equity += pnl
            trade_logs[-1].update({'exit_date': date, 'exit_price': p, 'pnl': pnl})
            position = 0
            entry_price = None
            equity -= cost_per_trade

        equity_curve.append(equity)
        position_log.append(position)

    equity = pd.Series(equity_curve, index=price.index)
    trades = pd.DataFrame(trade_logs)

    return equity, trades, position_log

def classification_backtest(df, probs_col, threshold=0.5):
    """Backtests a classification-based strategy using predicted probabilities"""

    df = df.copy()
    df['signal'] = (df[probs_col] >= threshold).astype(int)  # 1 or 0

    df['strategy_ret'] = df['signal'] * df['future_ret'] 
    df['equity'] = (1 + df['strategy_ret']).cumprod()

    return df