#!/usr/bin/env python3
"""Routing table: gains, bootstrap intervals, disagreement error and break-even per threshold."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import csm_rollouts as CR, csm_metrics as M                      # noqa: E402

SCRIPT_VERSION = "csm_routing_table/1.1"
SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5"]
FAMILY = {"tfidf_logreg": "linear", "gbm_percorrect": "gradient-boosted", "knn_valrate": "nearest-neighbor"}


def rates(cache, split):
    pids, succ, draws = CR.rate_matrix(cache, split=split, models=SEEDS)
    return pids, M.posterior_rates(succ, draws)                  # (M, n), members-only prior as in e7_routers


def base_rates(cache, split):
    _, succ, draws = CR.rate_matrix(cache, split=split, models=["base"])
    return M.posterior_rates(succ, draws)[0]                     # base-only prior, as in e7_routers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", required=True); ap.add_argument("--cache", required=True)
    ap.add_argument("--cache-val", required=True, help="validation cache the routers were fit on")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--band", type=float, nargs=2, default=[0.05, 0.30])
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = json.load(open(a.routers))
    tp, tr = rates(a.cache, "test"); vp, vr = rates(a.cache_val, "val"); tb = base_rates(a.cache, "test")
    n = len(tp); lo, hi = a.band
    m_val = {t: int(np.argmax([M.cover_at_tau(vr[m], t) for m in range(len(SEEDS))])) for t in a.taus}
    rng = np.random.default_rng(a.seed)
    boots = [rng.integers(0, n, n) for _ in range(a.n_boot)]
    out = {"script_version": SCRIPT_VERSION, "routers": a.routers, "cache": a.cache, "cache_val": a.cache_val,
           "n_test": n, "k_test": int(CR.rate_matrix(a.cache, split="test", models=SEEDS)[2].max()),
           "m_star_val_per_tau": {str(t): SEEDS[m] for t, m in m_val.items()}, "band": [lo, hi], "cells": [], "families": {}}
    for kind, fam in FAMILY.items():
        r = rep["routers"].get(kind)
        if not r or "per_init" not in r:
            continue
        inits = []
        for d in r["per_init"]:
            asg = np.asarray(d["assignment"], int)
            if len(asg) != n:
                print(f"FATAL: {kind} assignment has {len(asg)} prompts, cache test split has {n}"); return 1
            inits.append((d["seed"], asg))
        raucs = [M.restricted_auc(M.router_rates(tr, asg), lo, hi) for _, asg in inits]
        bi = [M.mixture_rates(np.vstack([tb, M.router_rates(tr, asg)])) for _, asg in inits]
        out["families"][fam] = {"rauc": float(np.median(raucs)), "rauc_min": float(min(raucs)), "rauc_max": float(max(raucs)),
                                "pass_at_1": float(np.median([np.mean(M.router_rates(tr, asg)) for _, asg in inits])),
                                "rauc_base_inclusive": float(np.median([M.restricted_auc(r, lo, hi) for r in bi])),
                                "pass_at_1_base_inclusive": float(np.median([np.mean(r) for r in bi]))}
        for t in a.taus:
            cov_best = max(M.cover_at_tau(tr[m], t) for m in range(len(SEEDS)))
            cov_or = M.cover_at_tau(M.oracle_rates(tr), t)
            L = cov_or - cov_best
            D = float(np.mean((tr.max(0) >= t) & (tr.min(0) < t)))
            gains = [M.cover_at_tau(M.router_rates(tr, asg), t) - cov_best for _, asg in inits]
            j = int(np.argsort(gains)[len(gains) // 2])            # the median initialization
            asg = inits[j][1]; rr = M.router_rates(tr, asg)
            bs = []
            for idx in boots:
                cb = max(np.mean(tr[m, idx] >= t) for m in range(len(SEEDS)))
                bs.append(np.mean(rr[idx] >= t) - cb)
            inD = (tr.max(0) >= t) & (tr.min(0) < t)
            eD = float(np.mean(rr[inD] < t)) if inD.any() else float("nan")
            out["cells"].append({"tau": t, "family": fam, "init": inits[j][0], "gain": gains[j],
                                 "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                                 "gain_oos": float(M.cover_at_tau(rr, t) - M.cover_at_tau(tr[m_val[t]], t)),
                                 "eD": eD, "breakeven_acc": 1 - L / D if D > 0 else float("nan"),
                                 "L_over_D": L / D if D > 0 else float("nan"), "D": D, "L": L,
                                 "spread": float(max(gains) - min(gains))})
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"run={SCRIPT_VERSION} routers={a.routers} cache={a.cache} n_test={n} k={out['k_test']} m_star_val={out['m_star_val_per_tau']}")
    print(f"{'tau':>5s} {'family':18s} {'gain':>7s} {'95% CI':>18s} {'gain_oos':>8s} {'eD':>6s} {'L/D':>6s} {'spread':>6s}")
    for c in out["cells"]:
        print(f"{c['tau']:5.2f} {c['family']:18s} {c['gain']:+7.3f} [{c['lo']:+6.3f},{c['hi']:+6.3f}] {c['gain_oos']:+8.3f} {c['eD']:6.3f} {c['L_over_D']:6.3f} {c['spread']:6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
