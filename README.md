# Project: Limits of Short-Horizon Predictability in Equity Prices

## Summary

This project studies whether short-horizon equity returns contain exploitable structure once realistic noise and trading frictions are taken inot account. We examine this through two complementary approaches:

1. **Rule-based mean reversion signals** derived from price deviations
2. **Feature-based classification models** trained to predict short-term returns

Across both approaches, apparent predictability is fragile and degrades rapidly under transaction costs, parameter perturbations, and regime changes. The results suggest structural limits to short-horizon predictability rather than deficiencies in model choice.

## Motivation

Short-horizon trading strategies often appear profitable in backtests, especially when evaulated without costs. However, at these horizons, microstructure noise, regime dependence, and transaction costs dominate small statistical edges.

The goal of this project is not to construct profitable alpha, but to understand:

- when predictability appears
- why it disappears,
- and how sensitive it is to modeling assumptions

This framing intentionally prioritizes insight over performance.

## Approach 

### 1. Mean Reversion Signals

We construct a simple mean reversion signal based on the Z-score of the price relative to a rolling mean and vol estimate. Trades are entered when price deviates sufficiently from its recent average and exited upon reversion.

Key characteristics:

- Single-asset, long/short
- No leverage, simple position logic
- Transactions costs applied symmetrically at entry and exit

This setup isolates the structural behavior of mean reversion without relying on complex modeling.

### 2. Feature-Based Classification

We also test whether short-horizon returns are predictable using standard features such as:

- recent returns,
- moving average ratios,
- volatility measures,
- price relative to rolling bands.

A simple Logistic Regression classifier is used to avoid overfitting. The classification model is evaluated both statiscally (ROC/AUC) and economically (backtest).

## Experimental Design

To avoid overstating results, we follow a deliberately conservative design:

- Strict train/test splits based on time
- Minimal hyperparameter tuning
- Senstivity analysis over key thresholds
- Transaction cost modeling
- Evaluation across multiple assets

No attempt is made to optimize parameters for peak backtest performance.

## Results

### Mean Reversion

- Mean reversion signals occasionally generate small profits before costs
- Even small transacations costs eliminate most apparent edge
- Drawdowns are large relative to cummulative returns
- Performance is highly sensitive to threshold choice and market regime

Overall, risk dominates reward at short horizons

### Classification

- Classification performance is weak (AUC close to random)
- Small predictive gains do not translate into stable PnL
- Strategy performance deteriorates quickly under threshold changes
- Added model complexity does not improve robustness

The classifier primarily learns noise rather than durable structure

## Key Insights

1. Apparent predictability at short horizons is fragile
2. Complexity does not rescue weak signal
3. Drawdowns dominate performance
4. Failure modes are structural, not implementation dependent

## Repository Structure

```graphql
core/
  signals.py        # signal/model construction
  features.py       # feature definitions
  backtest.py       # backtest engines
  metrics.py        # risk and performance metrics
  utils.py          # reusable utils (general, plotting, tabulation)

experiments/
  mean_reversion.ipynb
  classification.ipynb

archive/
  exploratory and unused experiments
```

## Takeaway

This project demonstrates that short-horizon equity predictability is limited not by modelling sophistication, but by noise, costs, and instability. Understanding why simple strategies fail is often more informative than constructing fragile successes

## Notes

This work is intended as an empirical study of predictability limits rather than a production trading system.
All results should be interpreted in that context.
