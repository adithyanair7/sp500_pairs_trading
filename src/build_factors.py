# src/pairs_selection.py
# Factor-aware pair selection with robust NaN handling and debug logging.
# - Align monthly factors to daily prices
# - Drop NA factors (bm, mom_12_1), monthly winsorize (2–98% by default)
# - Enforce min price history and column completeness
# - Compute correlations only for columns with >= min_overlap obs
# - Safe ADF/OLS guards
# - Silences "Mean of empty slice" RuntimeWarnings
#
# CLI:
#   python -m src.pairs_selection --config config.yaml [--debug]

import os
import sys
import warnings
import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.stattools import adfuller
from scipy import stats
from .utils import half_life

# Silence noisy NumPy warnings globally for this module
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0")
np.seterr(all="ignore")

def _ols_beta(a, b):
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 100:
        return np.nan
    y = x.iloc[:, 0].values
    X = x.iloc[:, 1].values
    if np.nanstd(X) == 0 or np.nanstd(y) == 0:
        return np.nan
    slope, intercept, r, p, se = stats.linregress(X, y)
    return slope

def _adf_p(series):
    x = series.dropna()
    if len(x) < 100:
        return np.nan
    if np.nanstd(x.values) == 0:
        return np.nan
    try:
        return adfuller(x, maxlag=1, regression='c', autolag='AIC')[1]
    except Exception:
        return np.nan

def _month_floor(dts):
    return pd.to_datetime(dts).values.astype('datetime64[M]')

def _winsorize_monthly(df_fac, cols, lo=0.02, hi=0.98):
    if 'month' not in df_fac.columns:
        return df_fac
    def clip_group(g):
        for c in cols:
            if c in g.columns:
                s = pd.to_numeric(g[c], errors='coerce')
                n = s.notna().sum()
                if n > 30:
                    ql = s.quantile(lo)
                    qh = s.quantile(hi)
                    g[c] = s.clip(ql, qh)
        return g
    # include_groups=False avoids pandas deprecation warnings
    return df_fac.groupby('month', group_keys=False).apply(clip_group, include_groups=False)

def _debug(msg, debug):
    if debug:
        print(msg, file=sys.stderr)

def run(cfg, debug=False):
    out_dir = cfg['outputs']['processed_dir']
    prices = pd.read_csv(os.path.join(out_dir, "prices.csv"), parse_dates=['date'])
    prices = prices.sort_values(['date','permno'])

    # --- Config ---
    sel = cfg['selection']
    start   = pd.to_datetime(cfg['universe']['start'])
    end     = pd.to_datetime(cfg['universe']['end'])
    min_px  = float(cfg['universe']['min_price'])
    lookback     = int(sel['lookback_days'])
    min_overlap  = int(sel['min_overlap_days'])
    corr_min     = float(sel['corr_min'])
    top_n        = int(sel['top_n'])
    factor_required = bool(sel.get('require_factors', True))
    lo_wins = float(sel.get('winsor_lo', 0.02))
    hi_wins = float(sel.get('winsor_hi', 0.98))

    # --- Price filters ---
    prices = prices[(prices['date'] >= start) & (prices['date'] <= end)].copy()
    prices = prices[prices['adj_close'] >= min_px].copy()
    prices['dollar_vol'] = prices['mcap'].abs() * prices['ret_d'].abs()
    prices = prices.groupby('permno').filter(
        lambda x: (x['dollar_vol'].median() >= cfg['universe']['min_dollar_vol'])
    )
    _debug(f"[DEBUG] after price/liquidity filters: rows={len(prices):,}, names={prices['permno'].nunique():,}", debug)

    # --- Factor-aware universe ---
    if factor_required:
        fac_path = os.path.join(out_dir, "factors.csv")
        if not os.path.exists(fac_path):
            raise FileNotFoundError(f"Missing {fac_path}. Run build_factors first or set selection.require_factors: false")
        fac = pd.read_csv(fac_path)
        fac.columns = [c.lower() for c in fac.columns]
        for c in ['month','permno','bm','mom_12_1']:
            if c not in fac.columns:
                raise ValueError(f"factors.csv missing required column '{c}'")

        fac['month'] = pd.to_datetime(fac['month'])
        # Drop NA and winsorize monthly
        fac = fac.dropna(subset=['bm','mom_12_1']).copy()
        fac = _winsorize_monthly(fac, cols=['bm','mom_12_1'], lo=lo_wins, hi=hi_wins)

        prices['month'] = _month_floor(prices['date'])
        fac_last = fac.sort_values(['permno','month']).drop_duplicates(['month','permno'], keep='last')
        pre = prices['permno'].nunique()
        prices = prices.merge(fac_last[['month','permno','bm','mom_12_1']], on=['month','permno'], how='inner')
        _debug(f"[DEBUG] merged factors: names {pre:,} -> {prices['permno'].nunique():,}, rows={len(prices):,}", debug)

    # --- Lookback window ---
    last_date = prices['date'].max()
    lb_start = last_date - pd.Timedelta(days=int(lookback * 1.2))  # buffer
    pnl = prices[prices['date'] >= lb_start].copy()
    # enforce overlap per permno
    pnl = pnl.groupby('permno').filter(lambda g: g['date'].nunique() >= min_overlap)
    _debug(f"[DEBUG] after lookback/overlap: rows={len(pnl):,}, names={pnl['permno'].nunique():,}", debug)

    # --- Panel construction ---
    px = pnl.pivot(index='date', columns='permno', values='adj_close').sort_index()

    # Keep only columns with at least min_overlap valid obs
    valid_counts = px.notna().sum(axis=0)
    keep_cols = valid_counts[valid_counts >= min_overlap].index
    px = px[keep_cols]
    _debug(f"[DEBUG] panel dates={len(px.index):,}, columns kept={px.shape[1]:,}", debug)

    if px.shape[1] < 2:
        raise RuntimeError("Not enough names after filters to form pairs. Loosen filters or extend lookback.")

    # Drop very sparse rows (days) if any — keep days where >= 2 names have data
    row_valid = px.notna().sum(axis=1)
    px = px[row_valid >= 2]
    _debug(f"[DEBUG] panel after row prune: dates={len(px.index):,}, columns={px.shape[1]:,}", debug)

    logpx = np.log(px)

    # Compute correlation only across the final kept columns; pandas will handle NaNs.
    # We already ensured each column has >= min_overlap non-NaNs.
    corr = logpx.corr(min_periods=min_overlap)

    # --- Pair selection ---
    candidates = []
    permnos = corr.columns.values
    n_cols = len(permnos)
    _debug(f"[DEBUG] computing pairs from {n_cols} names", debug)

    for i in range(n_cols):
        for j in range(i+1, n_cols):
            c = corr.iat[i, j]
            if not (pd.notna(c) and c >= corr_min):
                continue
            a = permnos[i]; b = permnos[j]
            ab = logpx[[a, b]].dropna()
            if len(ab) < min_overlap:
                continue
            # Guard against constant / near-constant slices
            if np.nanstd(ab[a].values) == 0 or np.nanstd(ab[b].values) == 0:
                continue
            beta = _ols_beta(ab[a], ab[b])
            if not np.isfinite(beta):
                continue
            spread = ab[a] - beta*ab[b]
            if spread.notna().sum() < min_overlap or np.nanstd(spread.values) == 0:
                continue
            p = _adf_p(spread)
            if pd.isna(p) or p > float(cfg['selection']['adf_alpha']):
                continue
            hl = half_life(spread)
            dsd = spread.diff().std(ddof=0)
            if not np.isfinite(dsd) or dsd == 0:
                continue
            candidates.append((
                int(a), int(b), float(beta), float(p),
                float(hl if np.isfinite(hl) else np.nan),
                float(dsd)
            ))

    pairs = pd.DataFrame(
        candidates,
        columns=['permno_a','permno_b','beta','adf_p','half_life','spread_vol']
    ).sort_values(['adf_p','spread_vol']).head(top_n)

    out_path = os.path.join(out_dir, "pairs.csv")
    pairs.to_csv(out_path, index=False)
    print(f"[pairs_selection] picked {len(pairs)} pairs -> {out_path}")
    return pairs

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dbg_cfg = bool(cfg.get('selection', {}).get('debug', False))
    run(cfg, debug=(args.debug or dbg_cfg))
