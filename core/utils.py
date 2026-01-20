import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from core.metrics import max_drawdown
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def fetch_data(tickers, start=None, end=None, period=None, interval='1d'):
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

def plot_equity(equity_dict, title="Equity Curve", fname=None):
    """ equity_dict: {label: pd.Series}"""
    plt.figure(figsize=(10, 5))
    for label, eq in equity_dict.items():
        plt.plot(eq.index, eq.values, label=label)
    plt.title(title)
    plt.ylabel("Equity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()

def plot_drawdowns(equity_tc, title="Drawdown", fname=None):
    """Drawdown plot"""
    drawdown, _ = max_drawdown(equity_tc)
    plt.figure(figsize=(10, 4))
    plt.plot(drawdown.index, drawdown.values, color='red')
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.grid(True)
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

def plot_price_z_position(price, z, position, entry_z, title="Price & Z-score", fname=None):
    """Price + Z-score + Position overlay"""
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax[0].plot(price.index, price.values)
    ax[0].set_title(title)

    ax[1].plot(z.index, z.values)
    ax[1].axhline(entry_z, linestyle="--", color="grey")
    ax[1].axhline(-entry_z, linestyle="--", color="grey")
    ax[1].axhline(0, color="black", linewidth=0.8)
    ax[1].fill_between(
        z.index, z.min(), z.max(),
        where=([True if pos !=0 else False for pos in position]),
        alpha=0.2
    )

    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()

def plot_trade_pnl(trades, title="Trade PnL Distribution", fname=None):
    """PnL distribution"""
    plt.figure(figsize=(8, 4))
    plt.hist(trades["pnl"], bins=30, alpha=0.7)
    plt.axvline(0, color="black", linestyle="--")
    plt.title(title)
    plt.xlabel("PnL")
    plt.ylabel("Count")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()

def plot_duration_vs_pnl(trades, title="Trade Duration vs PnL", fname=None):
    """PnL distribution"""
    durations = (trades["exit_date"] - trades["entry_date"]).dt.days

    plt.figure(figsize=(8, 5))
    plt.scatter(durations, trades["pnl"], alpha=0.6)
    plt.axhline(0, color="black", linestyle="--")
    plt.title(title)
    plt.xlabel("Duration (days)")
    plt.ylabel("PnL")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()

def plot_sensitivity(x, y, xlabel, ylabel, title="Sensitivity Analysis", fname=None):
    """Senstivity to params"""
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()

def plot_roc(y_true, y_prob, title="ROC Curve", fname=None):
    """ROC and AUC"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()

def strategy_summary(equity, trades, periods_per_year=252, tname=None):
    """Summarizes the MR strategy"""
    returns = equity.pct_change().dropna()

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(periods_per_year)
        if returns.std() > 0 else 0.0
    )

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    win_rate = (trades["pnl"] > 0).mean() if not trades.empty else 0.0
    avg_pnl = trades["pnl"].mean() if not trades.empty else 0.0
    n_trades = len(trades)

    df = pd.DataFrame([{
        "Total Return": total_return,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
        "Avg Trade PnL": avg_pnl,
        "Num Trades": n_trades
    }])

    if tname:
        df.to_markdown(tname, index=False)
    
    return df

def asset_comparison(results_dict, tname=None):
    """
    Compare results for different assets
    results_dict: {asset: (equity, trades)}
    """
    rows = []

    for asset, (equity, trades) in results_dict.items():
        returns = equity.pct_change().dropna()
        sharpe = (
            returns.mean() / returns.std() * np.sqrt(252)
            if returns.std() > 0 else 0.0
        )

        peak = equity.cummax()
        drawdown = (equity - peak) / peak

        rows.append({
            "Asset": asset,
            "Total Return": equity.iloc[-1] / equity.iloc[0] - 1,
            "Sharpe": sharpe,
            "Max Drawdown": drawdown.min(),
            "Num Trades": len(trades)
        })

    df = pd.DataFrame(rows)

    if tname:
        df.to_markdown(tname, index=False)
    
    return df

def cost_impact_table(no_cost_equity, cost_equity, tname=None):
    """Impact of including transaction costs"""
    
    df = pd.DataFrame([
        {
            "Scenario": "No Costs",
            "Total Return": no_cost_equity.iloc[-1] / no_cost_equity.iloc[0] - 1
        },
        {
            "Scenario": "With Costs",
            "Total Return": cost_equity.iloc[-1] / cost_equity.iloc[0] - 1
        }
    ])

    if tname:
        df.to_markdown(tname, index=False)

    return df

def trade_duration_stats(trades, tname=None):
    """Trade duration summary"""

    durations = (trades["exit_date"] - trades["entry_date"]).dt.days

    df = pd.DataFrame([{
        "Avg Duration (days)": durations.mean(),
        "Median Duration (days)": durations.median(),
        "Max Duration (days)": durations.max()
    }])

    if tname:
        df.to_markdown(tname, index=False)

    return df

def classification_metrics(y_true, y_pred, y_prob=None, tname=None):
    """Metrics for quality of classification"""

    row = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0)
    }

    if y_prob is not None:
        row["AUC"] = roc_auc_score(y_true, y_prob)

    df = pd.DataFrame([row])

    if tname:
        df.to_markdown(tname, index=False)
    
    return df

def threshold_sensitivity_table(thresholds, returns, tname=None):
    """Senstivity to thresholds"""
    
    df = pd.DataFrame({
        "Threshold": thresholds,
        "Strategy Return": returns
    })

    if tname:
        df.to_markdown(tname, index=False)
    
    return df