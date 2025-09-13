import yaml, math, itertools, subprocess, pandas as pd, os, json
from copy import deepcopy

CFG_PATH = "config.yaml"
OUT_DIR  = "./data/processed"
DAILY    = f"{OUT_DIR}/backtest_daily.csv"
PAIRS    = f"{OUT_DIR}/pairs.csv"

def run_backtest():
    subprocess.run(["python","-m","src.backtest","--config",CFG_PATH], check=True)

def metrics():
    d = pd.read_csv(DAILY, parse_dates=["date"])
    ret = d["pnl"].fillna(0.0)
    ann_mu = float(ret.mean() * 252.0)
    ann_sd = float(ret.std(ddof=0) * (252.0 ** 0.5))
    sharpe = float(ann_mu / ann_sd) if ann_sd > 0 else float("nan")
    pos_day = float((ret > 0).mean())
    empty_days = int((d.get("positions", 0) == 0).sum()) if "positions" in d.columns else None
    avg_pos = float(d.get("positions", pd.Series([float("nan")])).mean())
    avg_turn = float(d.get("gross_turnover", pd.Series([float("nan")])).mean())
    cfg = yaml.safe_load(open(CFG_PATH))
    tgt = float(cfg["risk"]["daily_vol_target"]) * (252.0 ** 0.5)
    rtt = float(ann_sd / max(tgt, 1e-12))
    return dict(ann_mu=ann_mu, ann_sd=ann_sd, sharpe=sharpe, pos_day=pos_day,
                empty_days=empty_days, avg_positions=avg_pos, avg_turnover=avg_turn,
                realized_to_target=rtt)

def oos_ok():
    if not os.path.exists(PAIRS):
        return True
    p = pd.read_csv(PAIRS, parse_dates=["beta_end"])
    if "beta_end" not in p.columns or p.empty:
        return True
    d = pd.read_csv(DAILY, parse_dates=["date"])
    return bool(d["date"].min() > p["beta_end"].max())

def write_cfg(base, patch):
    cfg = deepcopy(base)
    for k, sub in patch.items():
        cfg.setdefault(k, {})
        cfg[k].update(sub)
    yaml.safe_dump(cfg, open(CFG_PATH, "w"), sort_keys=False)

def patches_from_grid(grid_dict):
    keys, lists = [], []
    for key, arr in grid_dict.items():
        if not arr: continue
        if key=="risk2": key="risk"
        keys.append(key); lists.append(arr)
    if not lists:
        yield {}
        return
    import itertools
    for combo in itertools.product(*lists):
        patch = {}
        for k, sub in zip(keys, combo):
            patch.setdefault(k, {}).update(sub)
        yield patch

def main():
    base = yaml.safe_load(open(CFG_PATH))
    results = []

    grids = [
      # Primary risk/participation knobs
      {
        "risk":  [{"max_pair_vol_share": v} for v in [0.5, 0.7, 0.8, 0.9]] \
               + [{"vol_guard": g} for g in [0.75, 0.85, 1.00]],
        "backtest": [{"entry_z": z} for z in [2.4, 2.6, 2.8]] \
                  + [{"max_pairs_live": n} for n in [14, 16, 20]] \
                  + [{"per_name_cooldown_days": c} for c in [2, 3, 5]] \
                  + [{"per_name_limit": p} for p in [1, 2]],
        "enhancements": [{"reversion_eps": r} for r in [0.0, 0.05]] \
                      + [{"dsd_window": w} for w in [20, 30, 60]],
      },
      # Realism knobs
      {
        "risk": [{"vix_gate_on": x, "vix_max": m} for x,m in [(True,28),(True,35),(False,0)]] \
              + [{"pnl_clip_sigma": c} for c in [0.0, 3.0]],
      },
    ]

    target_sharpe_low, target_sharpe_high = 2.0, 2.3
    target_rtt_low, target_rtt_high = 0.85, 1.15

    def record(patch, met):
        row = {"patch": json.dumps(patch), **{k: (None if pd.isna(v) else float(v)) for k,v in met.items()}}
        row["oos_ok"] = bool(oos_ok())
        results.append(row)
        print(f"try: Sharpe={met['sharpe']:.2f}  σ={met['ann_sd']:.2%}  rtt={met['realized_to_target']:.2f}  "
              f"pos_day={met['pos_day']:.2f}  avg_pos={met['avg_positions']:.2f}  patch={patch}")
        return row

    done=False
    for grid in grids:
        for patch in patches_from_grid(grid):
            write_cfg(base, patch)
            try:
                run_backtest()
                met = metrics()
            except Exception as e:
                print("[skip]", e); continue
            row = record(patch, met)
            if (target_sharpe_low <= met["sharpe"] <= target_sharpe_high and
                target_rtt_low <= met["realized_to_target"] <= target_rtt_high and
                row["oos_ok"]):
                done=True; break
        if done: break

    pd.DataFrame(results).to_csv("tune_results.csv", index=False)
    print("\nSaved all tries to tune_results.csv")
    if results:
        best = max(results, key=lambda r: (r["sharpe"] if r["sharpe"] is not None else -1e9))
        print("Best so far:", json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
