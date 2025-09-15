# Advanced Pairs Trading on S&P 500

## Overview
This project implements a **bias-aware statistical arbitrage framework** for equity pairs trading within the S&P 500 universe. The framework emphasizes robust methodology and institutional-quality validation, avoiding common backtest pitfalls such as lookahead bias and unrealistic execution assumptions.

Out-of-sample (OOS) results over **2018–2024** demonstrate:
- **Sharpe Ratio (annualized):** ≈ **2.18**
- **Maximum Drawdown:** ≈ **−12.3%**
- **Annualized Return:** ≈ **13%**
- **Average Concurrent Positions:** ≈ **20–25**
- **Turnover:** ≈ **0.20**

These figures are consistent with a scalable, market-neutral statistical arbitrage strategy under realistic assumptions.

---

## Data Sources
All raw data was obtained through **Wharton Research Data Services (WRDS)**, ensuring institutional-grade quality and survivorship-bias–free coverage.

- **CRSP (Center for Research in Security Prices)**
  - Daily stock returns, prices, shares outstanding, and delisting adjustments.
  - Used to construct adjusted price series and unionized trading calendar.

- **Compustat (Fundamentals Quarterly – FUNDQ)**
  - Point-in-time financial statement variables.
  - Linked to CRSP securities via CCM linking table for factor construction.

- **Fama–French / Kenneth French Data Library**
  - Factor benchmarks (market, size, value, momentum) used as explanatory signals.

- **Proxy S&P 500 Membership Data**
  - Historical constituent information to approximate a dynamic S&P 500 universe over the backtest window.

This data foundation ensures the backtest is both **survivorship-bias free** and **point-in-time accurate**, consistent with academic and professional standards.

---

## Methodology
- **Signal Formation**
  - Engle–Granger two-step cointegration to identify long-run equilibrium relationships.
  - Standardized residuals (z-scores) used for entry/exit triggers with symmetric thresholds.

- **Execution Framework**
  - **t−1 signal execution**: positions decided at time *t−1*, executed at *t*.
  - **Unionized trading calendar** to synchronize across securities.
  - **Transaction cost modeling** including slippage and commissions.

- **Risk Management**
  - Volatility targeting to stabilize portfolio-level risk.
  - Per-name position caps to prevent concentration.
  - Dynamic throttling and optional drawdown guards to reduce exposure during adverse regimes.

---

## Repository Structure
- **`build_prices.py`** — Processes CRSP data into adjusted prices on a unionized calendar.
- **`build_factors.py`** — Constructs factor exposures from Compustat fundamentals and CRSP returns, exports `factors.csv`.
- **`pairs_selection.py`** — Applies Engle–Granger screening, computes hedge ratios, exports `pairs.csv`.
- **`backtest.py`** — Bias-aware engine with t−1 mechanics, costs, and risk controls. Exports `backtest_daily.csv` and `backtest_trades.csv`.
- **`run_pipeline.py`** — Orchestrates end-to-end workflow from raw WRDS data to final backtest results.
- **`utils.py`** — Shared functions for data alignment and calculations.
- **Artifacts**
  - `factors.csv` — Factor dataset.
  - `pairs.csv` — Selected pairs with p-values, scores, hedge ratios.
  - `backtest_daily.csv` — Daily returns, equity, and diagnostics.
  - `backtest_trades.csv` — Trade-level entries and exits.

---

## Results Summary

| Metric                | Value        |
|------------------------|--------------|
| Sharpe Ratio (annual) | ≈ **2.18**   |
| Max Drawdown          | ≈ **−12.3%** |
| Annualized Return     | ≈ **13%**    |
| Avg Breadth           | ≈ **20–25**  |
| Turnover              | ≈ **0.20**   |

### Equity Curve
![Equity Curve](notebooks/plots/final_equity.png)

---

## Notebooks
The `/notebooks/` directory contains detailed analyses and an academic-style report:
1. **01_Factors_Summary.ipynb** — Factor dataset inspection and correlations.
2. **02_Pairs_Selection_Summary.ipynb** — Cointegration screening diagnostics.
3. **03_Backtest_Results.ipynb** — Equity, drawdowns, monthly returns, trade distributions.
4. **04_Final_Report.ipynb** — Abstract, methodology, results, discussion, conclusion.

---

## Installation & Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline:
   ```bash
   python run_pipeline.py --config config.yaml
   ```
3. Explore results through the notebooks in `/notebooks/`.

---

## Conclusion
This project demonstrates the design and validation of a market-neutral statistical arbitrage framework with academically and professionally credible results. The use of **WRDS-sourced CRSP, Compustat, and factor data** ensures a survivorship-bias–free, point-in-time accurate dataset, making the backtest suitable for both academic research and professional evaluation.
