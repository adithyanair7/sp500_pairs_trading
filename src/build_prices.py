# src/build_prices.py
import os, pandas as pd, numpy as np, yaml
from .utils import smart_read, adj_close_from_crsp, compute_mcap

def run(cfg):
    out_dir = cfg['outputs']['processed_dir']
    os.makedirs(out_dir, exist_ok=True)

    # Try configured path + common fallbacks
    p = cfg['paths'].get('crsp_dsf')
    candidates = [p, "./data/raw/crsp_dsf.csv", "./data/raw/crsp.dsf.csv", "./data/raw/crsp.dsf",
                  "/Users/adi/Downloads/crsp_dsf.csv", "/Users/adi/Downloads/crsp.dsf.csv", "/Users/adi/Downloads/crsp.dsf"]
    cols = ['date','permno','prc','ret','shrout','cfacpr','cfacshr']
    df = smart_read(candidates, dtype={c:'float64' for c in ['prc','ret','shrout','cfacpr','cfacshr']}, usecols=lambda c: c in cols or c in ['permco','vol'])
    # Coerce & clean
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['permno','date'])
    # Ensure required columns present
    for col in ['cfacpr','cfacshr']:
        if col not in df: df[col] = 1.0
    # Build adj_close, ret_d, mcap
    df['adj_close'] = adj_close_from_crsp(df)
    if 'ret' in df and df['ret'].notna().any():
        df['ret_d'] = pd.to_numeric(df['ret'], errors='coerce')
    else:
        df['ret_d'] = df.groupby('permno')['adj_close'].apply(lambda s: s.pct_change())
    df['mcap'] = compute_mcap(df)
    out = df[['date','permno','adj_close','ret_d','mcap']].dropna(subset=['adj_close'])
    out.to_csv(os.path.join(out_dir, "prices.csv"), index=False)
    print(f"[build_prices] wrote {len(out):,} rows -> {os.path.join(out_dir, 'prices.csv')}")
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)
