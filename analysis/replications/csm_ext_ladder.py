#!/usr/bin/env python3
"""Selection, oracle and mixture summaries for one replication set (64 responses, new prompts, 3B)."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csm_rollouts as CR, csm_metrics as M                      # noqa: E402

SCRIPT_VERSION = "csm_ext_ladder/1.0"
SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5"]


def rates(cache, split):
    pids, succ, draws = CR.rate_matrix(cache, split=split, models=["base"] + SEEDS)
    R = M.posterior_rates(succ, draws)
    return pids, R[0], R[1:], int(draws.max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--cache-val", default=None); ap.add_argument("--routers", default=None)
    ap.add_argument("--band", type=float, nargs=2, default=[0.05, 0.30]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(); lo, hi = a.band
    pids, B, R, k = rates(a.cache, a.split)
    ra = lambda p: float(M.restricted_auc(p, lo, hi))
    p1 = lambda p: float(np.mean(p))
    n_seeds = R.shape[0]
    bps = [M.mixture_rates(np.vstack([B, R[m]])) for m in range(n_seeds)]
    ms = int(np.argmax([ra(p) for p in bps]))
    pool = M.mixture_rates(np.vstack([B, R]))
    comp_matched = M.mixture_rates(np.vstack([B, np.tile(R[ms], (n_seeds, 1))]))
    pb_raw = M.paired_bootstrap_gap(pool, bps[ms], lo, hi, n_boot=1500, seed=a.seed)
    pb_m = M.paired_bootstrap_gap(pool, comp_matched, lo, hi, n_boot=1500, seed=a.seed)
    out = {"script_version": SCRIPT_VERSION, "cache": a.cache, "split": a.split, "n": len(pids), "k": k, "band": [lo, hi],
           "ladder": {"base": [ra(B), p1(B)], "best_single_BplusP": [ra(bps[ms]), p1(bps[ms]), SEEDS[ms]],
                      "best_member_alone": [max(ra(R[m]) for m in range(n_seeds)), max(p1(R[m]) for m in range(n_seeds))],
                      "oracle_members": [ra(M.oracle_rates(R)), p1(M.oracle_rates(R))],
                      "oracle_with_base": [ra(M.mixture_rates(np.vstack([B, M.oracle_rates(R)]))), p1(M.mixture_rates(np.vstack([B, M.oracle_rates(R)])))],
                      "mixture_with_base": [ra(pool), p1(pool)],
                      "mixture_seed_only": [ra(M.mixture_rates(R)), p1(M.mixture_rates(R))]},
           "g_multi": {"raw": ra(pool) - ra(bps[ms]), "raw_ci": [pb_raw["ci_lo"], pb_raw["ci_hi"]],
                       "matched": ra(pool) - ra(comp_matched), "matched_ci": [pb_m["ci_lo"], pb_m["ci_hi"]],
                       "comparator": SEEDS[ms]}}
    if a.cache_val:
        _, _, Rv, _ = rates(a.cache_val, "val")
        mv = int(np.argmax(Rv.mean(axis=1)))
        cm_v = M.mixture_rates(np.vstack([B, np.tile(R[mv], (n_seeds, 1))]))
        pb_v = M.paired_bootstrap_gap(pool, cm_v, lo, hi, n_boot=1500, seed=a.seed)
        out["g_multi"].update({"matched_oos": ra(pool) - ra(cm_v), "matched_oos_ci": [pb_v["ci_lo"], pb_v["ci_hi"]],
                               "raw_oos": ra(pool) - ra(bps[mv]), "comparator_oos": SEEDS[mv]})
    if a.routers:
        rep = json.load(open(a.routers))
        fam = rep["routers"] if isinstance(rep.get("routers"), dict) else rep["families"]
        vals = {kk: v["rauc"] for kk, v in fam.items() if isinstance(v, dict) and "rauc" in v and kk not in ("best_single", "oracle")}
        best = max(vals, key=vals.get)
        out["ladder"]["routing_best_family"] = [vals[best], best]
        bi = {kk: v["rauc_base_inclusive"] for kk, v in fam.items() if isinstance(v, dict) and "rauc_base_inclusive" in v and kk not in ("best_single", "oracle")}
        if bi:
            bb = max(bi, key=bi.get)
            out["ladder"]["routing_best_family_with_base"] = [bi[bb], fam[bb].get("pass_at_1_base_inclusive"), bb]
    json.dump(out, open(a.out, "w"), indent=1)
    L = out["ladder"]; g = out["g_multi"]
    print(f"run={SCRIPT_VERSION} cache={a.cache} split={a.split} n={len(pids)} k={k} band=[{lo},{hi}]")
    print(f"  base {L['base'][0]:.3f} | best single {L['best_single_BplusP'][0]:.3f} ({L['best_single_BplusP'][2]}) | best member {L['best_member_alone'][0]:.3f} | oracle {L['oracle_members'][0]:.3f} (with base {L['oracle_with_base'][0]:.3f}/{L['oracle_with_base'][1]:.3f}) | mix+base {L['mixture_with_base'][0]:.3f} | mix seeds {L['mixture_seed_only'][0]:.3f}"
          + (f" | routing+base {L['routing_best_family_with_base'][0]:.3f}/{L['routing_best_family_with_base'][1]:.3f} ({L['routing_best_family_with_base'][2]})" if 'routing_best_family_with_base' in L else "")
          + (f" | routing {L['routing_best_family'][0]:.3f} ({L['routing_best_family'][1]})" if 'routing_best_family' in L else ""))
    print(f"  G raw {g['raw']:+.4f} [{g['raw_ci'][0]:+.3f},{g['raw_ci'][1]:+.3f}] matched {g['matched']:+.4f} [{g['matched_ci'][0]:+.3f},{g['matched_ci'][1]:+.3f}]"
          + (f" | OOS ({g['comparator_oos']}) matched {g['matched_oos']:+.4f} [{g['matched_oos_ci'][0]:+.3f},{g['matched_oos_ci'][1]:+.3f}] raw {g['raw_oos']:+.4f}" if 'matched_oos' in g else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
