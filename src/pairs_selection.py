# src/pairs_selection.py
# Snapshot-based, auditable pair selection with stability checks.
# - Fits beta/ADF ONLY on cfg['universe'] train window (no peeking)
# - Adds rolling-beta stability (beta_std) and subwindow stationarity (adf_pass_rate)
# - Enforces realistic quality gates + final per-name cap to boost breadth
#
# Output columns (at minimum): permno_a, permno_b, beta
# Useful extras: adf_p, adf_pass_rate, half_life, spread_vol, beta_start, beta_end, beta_nobs, beta_r2, beta_std

from __future__ import annotations
import os, math, warnings
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# Parallel (optional)
try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    _HAVE_PAR = True
except Exception:
    _HAVE_PAR = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

# ----------------------
# Utils
# ----------------------

def _nice(x): 
    return float(x) if (x is not None and np.isfinite(x)) else np.nan

def _ols_beta_with_intercept(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, int]:
    n = min(len(y), len(x))
    if n < 10: return np.nan, np.nan, 0
    X = np.c_[x[:n], np.ones(n)]
    coef, *_ = np.linalg.lstsq(X, y[:n], rcond=None)
    beta = float(coef[0])
    yhat = X @ coef
    ss_res = float(np.sum((y[:n] - yhat) ** 2))
    ss_tot = float(np.sum((y[:n] - y[:n].mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return beta, r2, n

def _adf_p(series: pd.Series) -> float:
    try:
        res = adfuller(series.dropna().values, maxlag=None, autolag="AIC")
        return float(res[1])
    except Exception:
        return np.nan

def _half_life(series: pd.Series) -> float:
    y = pd.Series(series).dropna()
    if len(y) < 30: return np.nan
    dy = y.diff().dropna()
    y_lag = y.shift(1).dropna().loc[dy.index]
    if len(dy) < 10 or len(dy) != len(y_lag): return np.nan
    X = np.c_[y_lag.values, np.ones(len(y_lag))]
    b, a = np.linalg.lstsq(X, dy.values, rcond=None)[0]  # dy = a + b*y_{t-1}
    k = -float(b)
    if not np.isfinite(k) or k <= 1e-6: return np.nan
    return float(math.log(2.0) / k)

def _rolling_beta_std(df: pd.DataFrame, win: int = 90, minp: int = 45) -> float:
    # fast beta_t via cov/var on logs; shift to be t-1 safe in spirit
    a, b = df.columns[:2]
    var_b = df[b].rolling(win, min_periods=minp).var(ddof=0).shift(1)
    cov_ab = df[a].rolling(win, min_periods=minp).cov(df[b]).shift(1)
    beta_t = cov_ab / var_b
    return float(beta_t.dropna().std(ddof=0)) if beta_t.notna().sum() >= 10 else np.nan

def _adf_pass_rate(spr: pd.Series, segments: int = 3, alpha: float = 0.05) -> float:
    spr = spr.dropna()
    T = len(spr)
    if T < 180 or segments < 2: 
        return np.nan
    idxs = np.linspace(0, T, segments+1).astype(int)
    passes = 0; valid = 0
    for i in range(segments):
        seg = spr.iloc[idxs[i]:idxs[i+1]]
        if len(seg) < 120: 
            continue
        p = _adf_p(seg)
        if np.isfinite(p):
            valid += 1
            if p <= alpha:
                passes += 1
    return float(passes/valid) if valid > 0 else np.nan

# ----------------------
# Worker
# ----------------------

def _pair_worker(args) -> Dict:
    (a, b, logpx_ab, train_start, train_end, adf_alpha, beta_roll_win) = args
    try:
        sub = logpx_ab.loc[train_start:train_end, [a, b]].dropna()
        if len(sub) < 160:
            return {}
        beta, r2, nobs = _ols_beta_with_intercept(sub[a].values, sub[b].values)
        if not np.isfinite(beta): 
            return {}
        spr = sub[a] - beta * sub[b]
        adf_full = _adf_p(spr)
        hl = _half_life(spr)
        dvol = float(spr.diff().dropna().std(ddof=0)) if spr.size else np.nan
        bstd = _rolling_beta_std(sub, win=int(beta_roll_win), minp=max(30, int(beta_roll_win*0.5)))
        pass_rate = _adf_pass_rate(spr, segments=3, alpha=adf_alpha)

        return {
            "permno_a": int(a),
            "permno_b": int(b),
            "beta": _nice(beta),
            "adf_p": _nice(adf_full),
            "adf_pass_rate": _nice(pass_rate),
            "half_life": _nice(hl),
            "spread_vol": _nice(dvol),
            "beta_start": str(pd.to_datetime(train_start).date()),
            "beta_end": str(pd.to_datetime(train_end).date()),
            "beta_nobs": int(nobs),
            "beta_r2": _nice(r2),
            "beta_std": _nice(bstd),
        }
    except Exception:
        return {}

# ----------------------
# Main selection
# ----------------------

def run(cfg: dict, debug: bool = False) -> pd.DataFrame:
    """
    Select pairs within cfg['universe'] (TRAIN window). No look-ahead.
    Requires: ./data/processed/prices.csv with columns date, permno, adj_close (+ optional volume/dollar_vol).
    """
    out_dir = cfg.get("outputs", {}).get("processed_dir", "./data/processed")
    ppath = os.path.join(out_dir, "prices.csv")
    if not os.path.exists(ppath):
        raise FileNotFoundError(f"Missing {ppath}. Build prices first.")

    train_start = pd.to_datetime(cfg["universe"]["start"])
    train_end   = pd.to_datetime(cfg["universe"]["end"])

    sel = cfg.get("selection", {})
    lookback_days     = int(sel.get("lookback_days", 504))
    min_overlap_days  = int(sel.get("min_overlap_days", 252))
    corr_min          = float(sel.get("corr_min", 0.85))
    adf_alpha         = float(sel.get("adf_alpha", 0.05))
    top_n             = int(sel.get("top_n", 200))                  # default ↑ for more breadth
    same_sector_only  = bool(sel.get("same_sector_only", True))
    n_jobs            = max(1, int(sel.get("n_jobs", max(1, (os.cpu_count() or 4)//2))))
    top_k_per_name    = int(sel.get("top_k_per_name", 10))          # candidate fan-out
    final_top_k_name  = int(sel.get("final_top_k_per_name", 3))     # final cap per name
    max_candidates    = int(sel.get("max_candidates", 300_000))
    beta_roll_win     = int(sel.get("beta_window", 90))             # for beta_std

    # Load prices in TRAIN window
    px = pd.read_csv(ppath, parse_dates=["date"]).sort_values(["date","permno"])
    px = px[(px["date"]>=train_start) & (px["date"]<=train_end)].copy()
    if px.empty:
        raise RuntimeError("No prices inside the requested train window.")

    # Liquidity guards
    min_price = float(cfg.get("universe",{}).get("min_price", 5.0))
    min_dollar = float(cfg.get("universe",{}).get("min_dollar_vol", 1e6))
    ok = px.groupby("permno")["adj_close"].min()
    keep_ids = set(ok[ok >= min_price].index.astype(int))
    # try dollar vol if available
    if "dollar_vol" in px.columns:
        dv = px.groupby("permno")["dollar_vol"].mean()
        keep_ids &= set(dv[dv >= min_dollar].index.astype(int))
    elif "volume" in px.columns:
        # crude proxy: avg( close * volume ) over train
        dv_est = (px["adj_close"] * px["volume"]).groupby(px["permno"]).mean()
        keep_ids &= set(dv_est[dv_est >= min_dollar].index.astype(int))
    px = px[px["permno"].isin(keep_ids)].copy()

    # Pivot to log prices
    wide = px.pivot(index="date", columns="permno", values="adj_close").sort_index()
    logpx = np.log(wide)

    # Optional sector restriction (if factors.csv exists)
    sector_map: Dict[int,int] = {}
    if same_sector_only:
        fpath = os.path.join(out_dir, "factors.csv")
        if os.path.exists(fpath):
            fac = pd.read_csv(fpath)
            gcol = "gsector" if "gsector" in fac.columns else ("sector" if "sector" in fac.columns else None)
            if gcol and "permno" in fac.columns:
                if "date" in fac.columns:
                    fac = fac[pd.to_datetime(fac["date"]).between(train_start, train_end)]
                elif "month" in fac.columns:
                    fac = fac[pd.to_datetime(fac["month"]).between(train_start, train_end)]
                sector_map = fac.dropna(subset=[gcol]).groupby("permno")[gcol].last().to_dict()

    # Correlation screen on last lookback_days inside TRAIN
    end_idx = logpx.index.max()
    start_idx = max(train_start, end_idx - pd.Timedelta(days=max(lookback_days, min_overlap_days)+30))
    corr_df = logpx.loc[start_idx:end_idx].copy()

    valid_cols = [c for c in corr_df.columns if corr_df[c].count() >= min_overlap_days]
    corr_df = corr_df[valid_cols]
    if corr_df.shape[1] < 2:
        return pd.DataFrame(columns=["permno_a","permno_b","beta"])

    corr_mat = corr_df.corr(min_periods=min_overlap_days)

    # Candidate list above corr_min (optionally same sector), limit fan-out per name
    cand: List[Tuple[int,int]] = []
    permnos = list(corr_df.columns.astype(int))
    for a in permnos:
        row = corr_mat.loc[a].drop(index=a, errors="ignore").dropna()
        if same_sector_only and sector_map:
            sa = sector_map.get(int(a), None)
            row = row[[b for b in row.index if sector_map.get(int(b), None) == sa]]
        row = row[row >= corr_min].sort_values(ascending=False)
        row = row.head(top_k_per_name)
        for b in row.index.astype(int):
            if a < b:
                cand.append((int(a), int(b)))
            if len(cand) >= max_candidates:
                break
        if len(cand) >= max_candidates:
            break

    if not cand:
        return pd.DataFrame(columns=["permno_a","permno_b","beta"])

    # Build tasks (train-only slices)
    tasks = []
    for a,b in cand:
        cols = [c for c in [a,b] if c in logpx.columns]
        if len(cols) < 2: 
            continue
        logpx_ab = logpx[[a,b]].copy()
        tasks.append((a,b,logpx_ab,train_start,train_end,adf_alpha,beta_roll_win))

    rows: List[Dict] = []
    if _HAVE_PAR and n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_pair_worker, t) for t in tasks]
            for fut in as_completed(futs):
                r = fut.result()
                if r and np.isfinite(r.get("beta", np.nan)):
                    rows.append(r)
    else:
        for t in tasks:
            r = _pair_worker(t)
            if r and np.isfinite(r.get("beta", np.nan)):
                rows.append(r)

    if not rows:
        return pd.DataFrame(columns=["permno_a","permno_b","beta"])

    df = pd.DataFrame(rows)

    # ----------------------
    # Quality gates (tight but sane)
    # ----------------------
    # Require enough data and decent fit
    df = df[df["beta_nobs"] >= min_overlap_days]
    df = df[df["beta_r2"] >= 0.80]

    # Stationarity must hold overall and in subsegments
    df = df[df["adf_p"] <= adf_alpha]
    df = df[df["adf_pass_rate"].fillna(0) >= 0.60]  # ≥60% of segments pass ADF

    # Mean-reversion speed and tradability
    df = df[df["half_life"].between(2.0, 25.0)]     # avoid glacial & ultra-fast
    df = df[df["spread_vol"].between(0.008, 0.060)] # avoid tiny and chaotic spreads

    # Hedge stability
    df = df[df["beta"].between(0.6, 1.6)]
    df = df[df["beta_std"].fillna(9e9) <= 0.30]     # rolling beta std ≤ 0.30

    if df.empty:
        return pd.DataFrame(columns=["permno_a","permno_b","beta"])

    # ----------------------
    # Scoring & final per-name cap
    # ----------------------
    eps = 1e-12
    score = (
        -np.log(df["adf_p"] + eps)
        + 0.8 * df["adf_pass_rate"].fillna(0)
        + 0.7 * (1.0 / df["half_life"].clip(lower=1e-3))
        - 0.6 * (df["beta"] - 1.0).abs()
        - 0.5 * df["beta_std"].fillna(0.5)
        - 0.1 * ((df["spread_vol"] - 0.02).abs() / 0.02)
    )
    df["score"] = score
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Final diversification: cap occurrences per name, then top_n
    keep = []
    seen = {}
    for _, r in df.iterrows():
        a, b = int(r["permno_a"]), int(r["permno_b"])
        if seen.get(a,0) >= final_top_k_name: 
            continue
        if seen.get(b,0) >= final_top_k_name:
            continue
        keep.append(r)
        seen[a] = seen.get(a,0)+1
        seen[b] = seen.get(b,0)+1

    df2 = pd.DataFrame(keep)
    if top_n > 0 and len(df2) > top_n:
        df2 = df2.head(top_n).copy()

    # Order / dtypes
    order_cols = ["permno_a","permno_b","beta","adf_p","adf_pass_rate","half_life","spread_vol",
                  "beta_start","beta_end","beta_nobs","beta_r2","beta_std"]
    for c in order_cols:
        if c not in df2.columns:
            df2[c] = np.nan
    return df2[order_cols].reset_index(drop=True)

# ----------------------
# CLI
# ----------------------
if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = run(cfg, debug=args.debug)
    out_dir = cfg.get("outputs", {}).get("processed_dir", "./data/processed")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pairs.csv")
    out.to_csv(path, index=False)
    print(f"[selector] wrote {len(out)} rows → {path}")
