# src/run_pipeline.py
import yaml, argparse
from .build_prices import run as build_prices
from .build_factors import run as build_factors
from .pairs_selection import run as select_pairs
from .backtest import run as backtest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print("[1/4] Building prices…")
    build_prices(cfg)
    print("[2/4] Building factors…")
    build_factors(cfg)
    print("[3/4] Selecting pairs…")
    select_pairs(cfg)
    print("[4/4] Backtesting…")
    backtest(cfg)
    print("[DONE] Outputs in", cfg['outputs']['processed_dir'])

if __name__ == "__main__":
    main()
