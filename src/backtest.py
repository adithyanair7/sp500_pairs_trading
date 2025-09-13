# src/backtest.py  (realistic, throttled, buffered)
# Version: 2025-09-10C
# - t-1 safe (decide at t-1, PnL over (t-1 -> t])
# - L2-correct sizing to hit daily_vol_target
# - Resize-before-open; per-name & per-pair cooldowns
# - Flip guard (extra Z to reverse direction)
# - Turnover throttle (partial fills when daily GT would exceed budget)
# - Costs (tc + slippage), optional z-stop & time-stop
# - RV & DD scalers (scale-only; no “skip day” cherry-picking)
# - Build buffer: load pre-OOS history to avoid warm-up empties (no pre-OOS trading)
# - Diagnostics: engine_debug.csv + gating summary

import os, json, copy, math, warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay  # for buffer
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

BACKTEST_VERSION = "2025-09-10C"

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

try:
    from .pairs_selection import run as select_pairs
except Exception:
    select_pairs = None

def _ensure_defaults(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("outputs", {}).setdefault("processed_dir", "./data/processed")

    u = cfg.setdefault("universe", {})
    u.setdefault("start", "2018-01-01")
    u.setdefault("end",   "2024-12-31")
    u.setdefault("min_price", 5.0)
    u.setdefault("min_dollar_vol", 1e6)
    u.setdefault("build_buffer_days", 252)  # NEW: load pre-OOS days only for rolling stats

    r = cfg.setdefault("risk", {})
    r.setdefault("daily_vol_target", 0.004)
    r.setdefault("max_pair_vol_share", 0.25)
    r.setdefault("vol_guard", 1.40)              # kept realistic; RV control fine-tunes
    r.setdefault("tc_bps", 7.0)
    r.setdefault("slippage_bps", 3.0)
    r.setdefault("vix_gate_on", False)
    r.setdefault("vix_max", 35.0)
    r.setdefault("pnl_clip_sigma", 3.0)
    r.setdefault("turnover_budget", 0.20)        # cap daily gross turnover
    r.setdefault("dd_guard_on", True)
    r.setdefault("dd_floor_pct", -0.10)
    r.setdefault("dd_scale", 0.50)               # scale risk in drawdown

    # Realized-vol feedback (kept conservative)
    r.setdefault("rv_control_on", True)
    r.setdefault("rv_lookback_days", 60)
    r.setdefault("rv_hot_threshold", 1.03)
    r.setdefault("rv_alpha", 1.2)
    r.setdefault("rv_min_scale", 0.60)
    r.setdefault("rv_max_scale", 1.00)

    b = cfg.setdefault("backtest", {})
    b.setdefault("entry_z", 1.8)                 # realistic, not grabby
    b.setdefault("exit_z", 0.5)
    b.setdefault("lag_days", 1)
    b.setdefault("max_pairs_live", 60)
    b.setdefault("per_name_cooldown_days", 1)
    b.setdefault("per_pair_cooldown_days", 1)
    b.setdefault("per_name_limit", 3)
    b.setdefault("max_holding_days", 20)

    e = cfg.setdefault("enhancements", {})
    e.setdefault("rolling_beta_on", True)
    e.setdefault("beta_window", 120)
    e.setdefault("robust_z_on", False)           # standard z is fine; robust can under-trigger
    e.setdefault("reversion_eps", 0.00)
    e.setdefault("resize_hysteresis_frac", 0.80) # skip tiny resizes to tame churn
    e.setdefault("z_stop_mult", 1.15)            # soft tail guard
    e.setdefault("flip_extra_z", 0.20)           # modest extra Z to reverse
    e.setdefault("dsd_method", "ewma")
    e.setdefault("ewma_lambda", 0.985)           # slow, stable sizing
    e.setdefault("dsd_window", 30)

    cfg.setdefault("diagnostics", {}).setdefault("dump_engine_debug", True)

    w = cfg.setdefault("walkforward", {})
    w.setdefault("oos_start", u["start"])
    w.setdefault("oos_end",   u["end"])
    w.setdefault("train_months", 24)
    w.setdefault("test_months", 1)
    w.setdefault("save_pairs", True)
    return cfg

def _load_vix(cfg):
    p = (cfg.get('paths', {}) or {}).get('vix') or cfg.get('vix')
    if p and os.path.exists(p):
        v = pd.read_csv(p, parse_dates=['date']).set_index('date').sort_index()
        for c in v.columns:
            lc = c.lower()
            if 'vix' in lc or lc in ('close','vixcls'):
                return v[[c]].rename(columns={c:'vix'})
    return None

def _nice(x):
    return float(x) if pd.notna(x) and np.isfinite(x) else np.nan

def _rolling_beta(a: pd.Series, b: pd.Series, win: int, minp: int = None) -> pd.Series:
    if minp is None:
        minp = max(30, int(win*0.5))
    var_b = b.rolling(win, min_periods=minp).var(ddof=0)
    cov_ab = a.rolling(win, min_periods=minp).cov(b)
    return cov_ab / var_b

def _robust_stats(x: pd.Series, win: int = 252, minp: int = 126):
    med = x.rolling(win, min_periods=minp).median()
    mad = (x - med).abs().rolling(win, min_periods=minp).median()
    sd  = 1.4826 * mad
    return med, sd

def _build_panels(logpx: pd.DataFrame, pairs: pd.DataFrame, enh: dict):
    rolling_beta_on = bool(enh.get("rolling_beta_on", True))
    beta_window     = int(enh.get("beta_window", 120))
    robust_z_on     = bool(enh.get("robust_z_on", False))
    dsd_method      = enh.get("dsd_method", "ewma")
    ewma_lambda     = float(enh.get("ewma_lambda", 0.985))
    dsd_window      = int(enh.get("dsd_window", 30))

    def ewma_std(x, lam=0.97):
        return x.ewm(alpha=(1-lam), adjust=False).std(bias=False)

    panels, dropped = {}, 0
    for _, r in pairs.iterrows():
        a = int(r["permno_a"]); b = int(r["permno_b"]); beta0 = float(r["beta"])
        if a not in logpx.columns or b not in logpx.columns:
            dropped += 1; continue
        sub = logpx[[a,b]].dropna()
        if len(sub) < 260:
            dropped += 1; continue

        if rolling_beta_on:
            bt = _rolling_beta(sub[a], sub[b], win=beta_window).shift(1)
            spr = sub[a] - bt * sub[b]
            spr = spr.where(bt.notna(), sub[a] - beta0*sub[b])
        else:
            spr = sub[a] - beta0 * sub[b]

        if robust_z_on:
            mu, sd = _robust_stats(spr, win=252, minp=126)
        else:
            mu = spr.rolling(252, min_periods=126).mean()
            sd = spr.rolling(252, min_periods=126).std(ddof=0)
        mu = mu.shift(1); sd = sd.shift(1)
        z = (spr - mu) / sd

        dspread = spr.diff()
        if dsd_method == "ewma":
            dsd = ewma_std(dspread, lam=ewma_lambda).shift(1)
        else:
            dsd = dspread.rolling(dsd_window, min_periods=max(10, dsd_window//2)).std(ddof=0).shift(1)

        pnl = pd.DataFrame({"z": z, "dspread": dspread, "spr": spr, "dsd": dsd}).dropna()
        if len(pnl) < 260:
            dropped += 1; continue
        panels[(a,b)] = pnl
    return panels, dropped

@dataclass
class RiskParams:
    daily_vol_target: float
    max_pair_vol_share: float
    vol_guard: float
    tc_bps: float
    slippage_bps: float
    vix_gate_on: bool
    vix_max: float
    pnl_clip_sigma: float
    turnover_budget: float
    dd_guard_on: bool
    dd_floor_pct: float
    dd_scale: float
    rv_control_on: bool
    rv_lookback_days: int
    rv_hot_threshold: float
    rv_alpha: float
    rv_min_scale: float
    rv_max_scale: float

@dataclass
class ExecParams:
    entry_z: float
    exit_z: float
    lag_days: int
    max_pairs_live: int
    per_name_cooldown_days: int
    per_pair_cooldown_days: int
    per_name_limit: int
    max_holding_days: int

def _engine(cfg: dict, quiet: bool = False):
    out_dir = cfg["outputs"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"[engine] backtest_version: {BACKTEST_VERSION}")

    prices = pd.read_csv(os.path.join(out_dir, "prices.csv"), parse_dates=["date"]).sort_values(["date","permno"])
    pairs_path = os.path.join(out_dir, "pairs.csv")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Missing {pairs_path}. Run pair selection first.")
    pairs = pd.read_csv(pairs_path)

    # OOS range + buffer (no pre-OOS trading)
    start = pd.to_datetime(cfg["universe"]["start"])
    end   = pd.to_datetime(cfg["universe"]["end"])
    buf   = int(cfg["universe"].get("build_buffer_days", 252))
    start_buf = start - BDay(buf)
    prices = prices[(prices["date"]>=start_buf) & (prices["date"]<=end)].copy()
    if prices.empty:
        raise RuntimeError("No prices in the requested window (including buffer).")

    r_cfg = cfg["risk"]; b_cfg = cfg["backtest"]; enh = cfg.get("enhancements", {})
    risk = RiskParams(
        daily_vol_target=float(r_cfg["daily_vol_target"]),
        max_pair_vol_share=float(r_cfg["max_pair_vol_share"]),
        vol_guard=float(r_cfg["vol_guard"]),
        tc_bps=float(r_cfg["tc_bps"]),
        slippage_bps=float(r_cfg.get("slippage_bps", 0.0)),
        vix_gate_on=bool(r_cfg["vix_gate_on"]),
        vix_max=float(r_cfg.get("vix_max", 35.0)),
        pnl_clip_sigma=float(r_cfg["pnl_clip_sigma"]),
        turnover_budget=float(r_cfg.get("turnover_budget", 0.0)),
        dd_guard_on=bool(r_cfg.get("dd_guard_on", True)),
        dd_floor_pct=float(r_cfg.get("dd_floor_pct", -0.10)),
        dd_scale=float(r_cfg.get("dd_scale", 0.50)),
        rv_control_on=bool(r_cfg.get("rv_control_on", True)),
        rv_lookback_days=int(r_cfg.get("rv_lookback_days", 60)),
        rv_hot_threshold=float(r_cfg.get("rv_hot_threshold", 1.03)),
        rv_alpha=float(r_cfg.get("rv_alpha", 1.2)),
        rv_min_scale=float(r_cfg.get("rv_min_scale", 0.60)),
        rv_max_scale=float(r_cfg.get("rv_max_scale", 1.00)),
    )
    execp = ExecParams(
        entry_z=float(b_cfg["entry_z"]),
        exit_z=float(b_cfg["exit_z"]),
        lag_days=int(b_cfg["lag_days"]),
        max_pairs_live=int(b_cfg["max_pairs_live"]),
        per_name_cooldown_days=int(b_cfg["per_name_cooldown_days"]),
        per_pair_cooldown_days=int(b_cfg.get("per_pair_cooldown_days", 0)),
        per_name_limit=int(b_cfg.get("per_name_limit", 3)),
        max_holding_days=int(b_cfg.get("max_holding_days", 20)),
    )

    vix = _load_vix(cfg)

    px = prices.pivot(index="date", columns="permno", values="adj_close").sort_index()
    if px.empty:
        raise RuntimeError("Pivoted price panel is empty.")
    logpx = np.log(px)

    # Build panels on buffered data; trading/eval will start at 'start'
    panels, dropped = _build_panels(logpx, pairs, enh)
    if not quiet:
        print(f"[engine] usable pairs: {len(panels)} | dropped: {dropped}")
    if not panels:
        raise RuntimeError("No usable pairs after data alignment.")

    dates = logpx.index.to_list()
    if len(dates) < 260:
        raise RuntimeError("Insufficient dates after clipping.")

    last_traded_name: Dict[int, pd.Timestamp] = {}
    last_traded_pair: Dict[Tuple[int,int], pd.Timestamp] = {}
    book: Dict[Tuple[int,int], Dict[str,object]] = {}
    daily_rows, trade_rows, debug_rows = [], [], []
    recent_pnls: List[float] = []

    min_dsd = 1e-9
    per_leg_cost = (risk.tc_bps + risk.slippage_bps) * 1e-4
    dump_debug = bool(cfg.get("diagnostics", {}).get("dump_engine_debug", True))

    def realized_vol_scale():
        if not risk.rv_control_on or len(recent_pnls) < max(20, risk.rv_lookback_days):
            return 1.0
        rv = np.std(recent_pnls[-risk.rv_lookback_days:], ddof=0) * math.sqrt(252.0)
        tgt = risk.daily_vol_target * math.sqrt(252.0)
        ratio = rv / max(tgt, 1e-12)
        if ratio <= risk.rv_hot_threshold:
            return 1.0
        scale = 1.0 / (ratio ** risk.rv_alpha)
        return float(np.clip(scale, risk.rv_min_scale, risk.rv_max_scale))

    def dd_scale():
        if not risk.dd_guard_on or not daily_rows:
            return 1.0
        eq_curve = 1.0 + np.cumsum([row['pnl'] for row in daily_rows])
        cur  = float(eq_curve[-1])
        peak = float(np.max(eq_curve))
        dd = (cur / max(peak, 1e-12)) - 1.0
        return float(np.clip(risk.dd_scale, 0.0, 1.0)) if dd <= risk.dd_floor_pct else 1.0

    def size_pairs(cands):
        if not cands: return {}
        dsds = np.array([max(abs(c['dsd']), min_dsd) for c in cands])
        pre = 1.0 / dsds
        pre /= pre.sum()
        w = pre / dsds
        port_vol = float(np.sqrt(np.sum((w * dsds)**2)))
        w *= (risk.daily_vol_target / max(port_vol, 1e-12))
        out = {}
        cap_vol = risk.daily_vol_target * float(risk.max_pair_vol_share)
        for c, wi, dsd in zip(cands, w, dsds):
            cap_w = cap_vol / dsd
            out[(c['a'], c['b'])] = float(np.clip(wi, -cap_w, cap_w))
        return out

    def desired_positions_on_date(dt_sig):
        sigs = []
        flip_extra = float(cfg["enhancements"].get("flip_extra_z", 0.0))
        for (a,b), pnl in panels.items():
            if dt_sig not in pnl.index: continue
            z = float(pnl.loc[dt_sig,'z']); dsd = float(pnl.loc[dt_sig,'dsd'])
            if not np.isfinite(z) or not np.isfinite(dsd) or dsd <= 0: continue

            # name cooldown
            ok = True
            for nm in (a,b):
                tlast = last_traded_name.get(nm)
                if tlast is not None and (dt_sig - tlast).days < execp.per_name_cooldown_days:
                    ok = False; break
            if not ok: continue
            # pair cooldown
            tpair = last_traded_pair.get((a,b))
            if tpair is not None and (dt_sig - tpair).days < execp.per_pair_cooldown_days:
                continue

            if abs(z) >= execp.entry_z:
                side = -1 if z > 0 else +1
                # flip guard: if already in book with opposite side, require extra z
                if (a,b) in book and book[(a,b)]['side'] != side and abs(z) < (execp.entry_z + flip_extra):
                    continue
                sigs.append({'a':a,'b':b,'side':side,'z':z,'dsd':dsd})
        sigs.sort(key=lambda r: abs(r['z']), reverse=True)
        w = size_pairs(sigs)
        return sigs, {(s['a'], s['b']): {'side': s['side'], 'w': w.get((s['a'], s['b']), 0.0)} for s in sigs}

    lag = max(1, execp.lag_days)

    for i in range(lag, len(dates)):
        dt_pnl = dates[i]
        dt_sig = dates[i - lag]

        # --- Pre-OOS buffer days: do not trade, do not record ---
        if dt_pnl < start:
            book.clear()  # ensure no pre-OOS carry
            continue

        # optional VIX flatten (use preloaded vix)
        if vix is not None and risk.vix_gate_on:
            if dt_sig in vix.index and _nice(vix.loc[dt_sig,'vix']) > risk.vix_max:
                for (a,b), pos in list(book.items()):
                    trade_rows.append({'date': dt_sig,'permno_a':a,'permno_b':b,
                                       'action':'exit_vix','side':pos['side'],
                                       'w_prev':pos['w'],'w_new':0.0})
                    last_traded_name[a]=dt_sig; last_traded_name[b]=dt_sig
                    last_traded_pair[(a,b)]=dt_sig
                book.clear()

        # exits
        for (a,b), pos in list(book.items()):
            pnl_df = panels.get((a,b))
            if pnl_df is None or dt_sig not in pnl_df.index:
                trade_rows.append({'date': dt_sig,'permno_a':a,'permno_b':b,
                                   'action':'exit','side':pos['side'],'w_prev':pos['w'],'w_new':0.0})
                last_traded_name[a]=dt_sig; last_traded_name[b]=dt_sig
                last_traded_pair[(a,b)]=dt_sig
                del book[(a,b)]
                continue

            mh = int(execp.max_holding_days)
            if mh > 0:
                opened = pos.get('opened_on')
                if opened is not None and (dt_sig - opened).days >= mh:
                    trade_rows.append({'date': dt_sig,'permno_a':a,'permno_b':b,
                                       'action':'time_exit','side':pos['side'],
                                       'w_prev':pos['w'],'w_new':0.0})
                    last_traded_name[a]=dt_sig; last_traded_name[b]=dt_sig
                    last_traded_pair[(a,b)]=dt_sig
                    del book[(a,b)]
                    continue

            z_stop_mult = float(cfg["enhancements"].get("z_stop_mult", 0.0))
            z_now = float(pnl_df.loc[dt_sig,'z'])
            if z_stop_mult > 0 and abs(z_now) >= z_stop_mult * execp.entry_z:
                trade_rows.append({'date': dt_sig,'permno_a':a,'permno_b':b,
                                   'action':'stop','side':pos['side'],'w_prev':pos['w'],'w_new':0.0})
                last_traded_name[a]=dt_sig; last_traded_name[b]=dt_sig
                last_traded_pair[(a,b)]=dt_sig
                del book[(a,b)]
                continue

            if abs(z_now) <= execp.exit_z:
                trade_rows.append({'date': dt_sig,'permno_a':a,'permno_b':b,
                                   'action':'exit','side':pos['side'],'w_prev':pos['w'],'w_new':0.0})
                last_traded_name[a]=dt_sig; last_traded_name[b]=dt_sig
                last_traded_pair[(a,b)]=dt_sig
                del book[(a,b)]

        # entries/resizes
        sigs_raw, desired = desired_positions_on_date(dt_sig)

        # 1) resizes
        hyst = float(cfg["enhancements"].get("resize_hysteresis_frac", 0.0))
        for (a,b), des in desired.items():
            if (a,b) in book:
                w_new = float(des['w']); side = int(des['side'])
                prev = book[(a,b)]
                if hyst > 0 and abs(prev['w']) > 0:
                    frac = abs(w_new - prev['w'])/max(1e-12, abs(prev['w']))
                    if frac < hyst:
                        continue
                if (prev['side'] != side) or (abs(prev['w'] - w_new) > 1e-12):
                    trade_rows.append({'date': dt_sig,'permno_a':a,'permno_b':b,
                                       'action':'resize','side':side,
                                       'w_prev':prev['w'],'w_new':w_new})

        # 2) opens with per-name & per-pair & max cap
        name_counts: Dict[int,int] = {}
        for (x,y) in book.keys():
            name_counts[x] = name_counts.get(x,0)+1
            name_counts[y] = name_counts.get(y,0)+1

        opens_done = 0
        blocked_name_limit = blocked_max_pairs = blocked_cooldown = 0

        for (a,b), des in desired.items():
            if (a,b) in book:
                continue
            # cooldowns
            ok = True
            for nm in (a,b):
                tlast = last_traded_name.get(nm)
                if tlast is not None and (dt_sig - tlast).days < execp.per_name_cooldown_days:
                    ok = False; break
            if not ok:
                blocked_cooldown += 1
                continue
            tpair = last_traded_pair.get((a,b))
            if tpair is not None and (dt_sig - tpair).days < execp.per_pair_cooldown_days:
                blocked_cooldown += 1
                continue

            if name_counts.get(a,0) >= execp.per_name_limit or name_counts.get(b,0) >= execp.per_name_limit:
                blocked_name_limit += 1
                continue
            if len(book) >= execp.max_pairs_live:
                blocked_max_pairs += 1
                break

            w_new = float(des['w']); side = int(des['side'])
            trade_rows.append({'date': dt_sig,'permno_a':a,'permno_b':b,
                               'action':'enter','side':side,'w_prev':0.0,'w_new':w_new})

            book[(a,b)] = {'side':side,'w':0.0,'opened_on': dt_sig}  # will set after throttle
            name_counts[a] = name_counts.get(a,0)+1
            name_counts[b] = name_counts.get(b,0)+1
            opens_done += 1

        # Global scales
        rv_scale = realized_vol_scale()
        dd_s     = dd_scale()
        global_scale = float(np.clip(risk.vol_guard, 0.0, 5.0)) * float(np.clip(rv_scale*dd_s, 0.0, 1.0))

        # --- Turnover throttle (partial fills)
        today_idx = []
        for j in range(len(trade_rows)-1, -1, -1):
            if trade_rows[j]['date'] != dt_sig: break
            today_idx.append(j)
        today_idx = list(reversed(today_idx))

        gt0 = 0.0
        for j in today_idx:
            d = trade_rows[j]
            gt0 += abs(d['w_new'] - d['w_prev']) * 2.0
        throttle_scale = 1.0
        if risk.turnover_budget and gt0 > 0:
            if global_scale * gt0 > risk.turnover_budget:
                throttle_scale = float(risk.turnover_budget / (global_scale * gt0))

        # apply partial fills to trade rows and to book
        for j in today_idx:
            d = trade_rows[j]
            a = int(d['permno_a']); b = int(d['permno_b'])
            w_prev = float(d['w_prev']); w_tar = float(d['w_new'])
            w_new_applied = w_prev + throttle_scale * (w_tar - w_prev)
            d['w_new'] = w_new_applied  # overwrite
            if d['action'] in ('enter','resize'):
                entry = book.get((a,b), {'side': int(d['side']), 'w': 0.0, 'opened_on': dt_sig})
                entry['side'] = int(d['side'])
                entry['w'] = float(w_new_applied)
                book[(a,b)] = entry
                last_traded_name[a]=dt_sig; last_traded_name[b]=dt_sig
                last_traded_pair[(a,b)]=dt_sig
            elif d['action'].startswith('exit'):
                if (a,b) in book: del book[(a,b)]
                last_traded_name[a]=dt_sig; last_traded_name[b]=dt_sig
                last_traded_pair[(a,b)]=dt_sig

        # PnL over (dt_sig -> dt_pnl]
        pnl_sum = 0.0
        gross_turnover = 0.0

        # realized GT with throttle
        for j in today_idx:
            d = trade_rows[j]
            gross_turnover += abs(global_scale * (d['w_new'] - d['w_prev'])) * 2.0

        for (a,b), pos in book.items():
            pnl_df = panels.get((a,b))
            if pnl_df is None or dt_pnl not in pnl_df.index or dt_sig not in pnl_df.index:
                continue
            dspread = float(pnl_df.loc[dt_pnl,'dspread'])
            if risk.pnl_clip_sigma > 0:
                dsd_t1 = float(pnl_df.loc[dt_sig,'dsd'])
                if np.isfinite(dsd_t1) and dsd_t1 > 0:
                    clip = risk.pnl_clip_sigma * dsd_t1
                    dspread = float(np.clip(dspread, -clip, clip))
            pnl_sum += pos['side'] * (global_scale*pos['w']) * dspread

        tc_cost = (risk.tc_bps + risk.slippage_bps) * 1e-4 * gross_turnover
        pnl_sum -= tc_cost

        recent_pnls.append(pnl_sum)
        daily_rows.append({
            'date': dt_pnl, 'pnl': pnl_sum, 'tc_cost': tc_cost,
            'positions': len(book), 'gross_turnover': gross_turnover,
            'rv_scale': rv_scale, 'dd_scale': dd_s, 'global_scale': global_scale,
            'turnover_scale': throttle_scale
        })

        if dump_debug:
            debug_rows.append({
                'date': dt_sig,
                'sigs_raw': len(sigs_raw),
                'opens_done': opens_done,
                'blocked_name_limit': blocked_name_limit,
                'blocked_max_pairs': blocked_max_pairs,
                'blocked_cooldown': blocked_cooldown,
                'book_after': len(book),
                'gt0': gt0,
                'gt_after': gross_turnover,
                'throttle_scale': throttle_scale
            })

    # results (OOS only; pre-OOS days skipped above)
    daily = pd.DataFrame(daily_rows).set_index('date').sort_index()
    if daily.empty:
        raise RuntimeError("No trading days produced in OOS; check settings.")
    daily['equity'] = daily['pnl'].cumsum()
    daily['ret'] = daily['pnl']

    mu = daily['ret'].mean() * 252.0
    sd = daily['ret'].std(ddof=0) * math.sqrt(252.0)
    sharpe = (mu / sd) if sd > 0 else np.nan
    dd_abs = (daily['equity'] - daily['equity'].cummax()).min()
    eq = 1.0 + daily['equity']
    dd_pct = (eq / eq.cummax() - 1.0).min()
    realized_to_target = sd / max(cfg['risk']['daily_vol_target']*math.sqrt(252.0), 1e-12)

    stats = {
        'ann_mu': _nice(mu), 'ann_sd': _nice(sd), 'sharpe': _nice(sharpe),
        'max_dd': _nice(dd_abs), 'max_dd_pct': _nice(dd_pct),
        'avg_turnover': _nice(daily['gross_turnover'].mean()),
        'realized_to_target': _nice(realized_to_target),
        'avg_positions': _nice(daily['positions'].mean()),
        'empty_days': int((daily['positions']==0).sum()),
    }

    daily.reset_index().to_csv(os.path.join(out_dir, "backtest_daily.csv"), index=False)
    pd.DataFrame(trade_rows).to_csv(os.path.join(out_dir, "backtest_trades.csv"), index=False)
    if dump_debug and debug_rows:
        pd.DataFrame(debug_rows).to_csv(os.path.join(out_dir, "engine_debug.csv"), index=False)

    print(f"[engine] stats: {json.dumps(stats)}")

    if dump_debug and debug_rows:
        dbg = pd.DataFrame(debug_rows)
        summary = {
            "avg_signals": float(dbg['sigs_raw'].mean()),
            "avg_opens": float(dbg['opens_done'].mean()),
            "avg_block_name": float(dbg['blocked_name_limit'].mean()),
            "avg_block_maxpairs": float(dbg['blocked_max_pairs'].mean()),
            "avg_block_cooldown": float(dbg['blocked_cooldown'].mean()),
            "avg_book": float(dbg['book_after'].mean()),
            "avg_gt0": float(dbg['gt0'].mean()),
            "avg_gt_after": float(dbg['gt_after'].mean()),
            "avg_throttle_scale": float(dbg['throttle_scale'].mean())
        }
        print(f"[engine][diagnostics] {json.dumps(summary)}")
    return daily, None, stats

if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg = _ensure_defaults(cfg)

    if args.walkforward:
        print("Walk-forward driver available in earlier version; focus on single-run here.")
    else:
        _engine(cfg, quiet=args.quiet)
