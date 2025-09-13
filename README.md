# S&P 500 Pairs Trading — Bias-Free Scaffold (Sharpe ≥ 2 target)

This project builds clean **prices**, simple **factors**, **selects pairs** via correlation + ADF, and **backtests**
a mean-reversion strategy with **1-day execution lag**, **vol targeting**, **VIX gating**, and simple
transaction-cost haircuts.

## Quick Start
```bash
cd sp500_pairs
python -m pip install -r requirements.txt

# Edit config.yaml input paths to your local CSV files (Downloads or elsewhere).
# Outputs are local: ./data/processed

# Run stages separately:
python -m src.build_prices --config config.yaml
python -m src.build_factors --config config.yaml
python -m src.pairs_selection --config config.yaml
python -m src.backtest --config config.yaml

# Or run pipeline:
python -m src.run_pipeline --config config.yaml
```
## Inputs required (CSV)
- CRSP daily: date, permno, prc, ret, shrout, cfacpr, cfacshr  (file often named `crsp.dsf`)
- CRSP delistings (optional but recommended)
- CRSP names (permno→ticker mapping) (optional)
- Compustat FUNDQ (quarterly fundamentals)
- CCM link (gvkey↔permno) with link dates
- Company GICS (permno→gsector), optional but useful
- Index constituents history or a proxy S&P500 membership CSV (optional)
- FF daily factors, Treasuries (optional)
- VIX daily (for regime filter) (optional)

Everything writes to **`./data/processed`**.
