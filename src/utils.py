# src/utils.py
import os, pandas as pd, numpy as np

def smart_read(path_or_candidates, **kwargs):
    """
    Accept a single string path or a list of candidates. Returns a DataFrame.
    Tries candidates in order and returns the first that exists.
    """
    candidates = path_or_candidates if isinstance(path_or_candidates, (list, tuple)) else [path_or_candidates]
    for p in candidates:
        if p and os.path.exists(p):
            return pd.read_csv(p, **kwargs)
    raise FileNotFoundError(f"None of the paths exist: {candidates}")

def winsorize(s, lo=0.01, hi=0.99):
    if s.isna().all():
        return s
    ql, qh = s.quantile(lo), s.quantile(hi)
    return s.clip(ql, qh)

def month_floor(s):
    return pd.to_datetime(s).values.astype('datetime64[M]')

def compute_mcap(df):
    # CRSP prc is signed; shares in thousands.
    cfacpr = df['cfacpr'].replace({0:np.nan}).fillna(1.0)
    cfacshr = df['cfacshr'].replace({0:np.nan}).fillna(1.0)
    px = np.abs(df['prc'])
    shr = (df['shrout'] * 1000.0)
    return px * shr * (cfacpr / cfacshr)

def adj_close_from_crsp(df):
    cfacpr = df['cfacpr'].replace({0:np.nan}).fillna(1.0)
    cfacshr = df['cfacshr'].replace({0:np.nan}).fillna(1.0)
    return np.abs(df['prc']) / cfacpr * cfacshr

def zscore_series(x, win):
    m = x.rolling(win, min_periods=max(10, win//5)).mean()
    s = x.rolling(win, min_periods=max(10, win//5)).std(ddof=0)
    return (x - m) / s

def half_life(series):
    x = series.dropna()
    if len(x) < 50:
        return np.nan
    y = x.diff().dropna()
    x1 = x.shift(1).dropna().reindex_like(y)
    if len(y) < 30:
        return np.nan
    # OLS slope b in y = a + b*x_{t-1}
    n = len(y)
    x1c = x1 - x1.mean()
    yc  = y - y.mean()
    denom = (x1c*x1c).sum()
    if denom == 0:
        return np.nan
    b = (x1c*yc).sum() / denom
    if (1+b) <= 0:
        return np.nan
    import math
    return -np.log(2)/np.log(1+b)

def annualize_sharpe(daily_ret):
    mu = daily_ret.mean()*252.0
    sd = daily_ret.std(ddof=0)*np.sqrt(252.0)
    return 0.0 if sd==0 else mu/sd
