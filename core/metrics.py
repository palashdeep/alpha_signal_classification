import numpy as np

def sharpe_ratio(returns, periods_per_year=252):
    """Calculate the annualized Sharpe Ratio of a return series."""
    if returns.std() == 0:
        return 0
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)

def cagr(equity, periods_per_year=252):
    """Calculate CAGR from an equity series."""
    n = len(equity)
    if n == 0:
        return 0.0
    total_return = equity[-1] / equity[0]
    years = n / periods_per_year
    return total_return ** (1 / years) - 1

def max_drawdown(equity):
    """Calculate the maximum drawdown from an equity series."""
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown, drawdown.min()

def calmar_ratio(returns):
    """Calculate the Calmar Ratio from a return series."""
    cagr = cagr(returns)
    _, max_dd = max_drawdown(returns)
    if max_dd == 0:
        return 0
    return cagr / abs(max_dd)

def win_rate(trades):
    """Calculate the win rate from a DataFrame of trades."""
    if trades.empty:
        return 0
    wins = trades[trades['pnl'] > 0]
    return len(wins) / len(trades)

