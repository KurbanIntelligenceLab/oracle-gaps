"""Place the voting arms on the rAUC scale beside the single-response baselines."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csm_rollouts, csm_metrics as M  # noqa: E402

SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5"]
VOTES = [1, 3, 5, 7, 9]
TAUS = [0.05, 0.10, 0.20, 0.30]   # overridden by --taus


def rates(cache, split, models):
    pids, succ, draws = csm_rollouts.rate_matrix(cache, split=split, models=models)
    R = np.asarray(M.posterior_rates(succ, draws))
    return list(pids), {m: R[i] for i, m in enumerate(models)}


def summarise(r, lo, hi):
    return {"rauc": float(M.restricted_auc(r, lo, hi)),
            **{f"cov@{t:.2f}": float(M.cover_at_tau(r, t)) for t in TAUS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", default="analysis/cache_sc_vote")
    ap.add_argument("--pool", default="analysis/cache_m5pool")
    ap.add_argument("--tau-lo", type=float, default=0.05)
    ap.add_argument("--tau-hi", type=float, default=0.30)
    ap.add_argument("--out", default="analysis/sc_vote_results.json")
    ap.add_argument("--taus", nargs="+", type=float, default=None)
    a = ap.parse_args()
    lo, hi = a.tau_lo, a.tau_hi
    if a.taus:
        TAUS[:] = a.taus
    res = {"script": "csm_sc_analyze/1.0", "band": [lo, hi], "votes": VOTES}
    sc_models = [f"{arm}_v{v}" for arm in ["scpool", "scpoolb"] + [f"sc{s}" for s in SEEDS] for v in VOTES]
    for split in ("val", "test"):
        p_pids, P = rates(a.pool, split, ["base"] + SEEDS)
        s_pids, S = rates(a.sc, split, sc_models)
        common = [p for p in p_pids if p in set(s_pids)]
        pi = {p: i for i, p in enumerate(p_pids)}; si = {p: i for i, p in enumerate(s_pids)}
        P = {m: v[[pi[p] for p in common]] for m, v in P.items()}
        S = {m: v[[si[p] for p in common]] for m, v in S.items()}
        # Two scales. The pool cache holds 16-draw beta-binomial POSTERIORS; the
        # vote cache holds 4000-trial Monte-Carlo rates, which are unshrunk. A
        # comparison across the two mixes estimators, so the PRIMARY reference
        # is rebuilt on the vote scale from each seed's own v=1 arm, which is
        # that seed's raw 16-draw rate up to Monte-Carlo noise. The posterior
        # scale is kept for context only.
        Rpost = np.vstack([P[s] for s in SEEDS])
        R = np.vstack([S[f"sc{s}_v1"] for s in SEEDS])
        orac, mix = M.oracle_rates(R), M.mixture_rates(R)
        best_i = int(np.argmax([M.restricted_auc(R[i], lo, hi) for i in range(len(SEEDS))]))
        best = R[best_i]
        row = {"n_prompts": len(common), "best_single": SEEDS[best_i],
               "scale_note": "reference arms rebuilt from the seeds' v=1 vote arms (raw scale)",
               "check_scpool_v1_vs_mixture_of_v1_maxabs": float(np.abs(S["scpool_v1"] - mix).max()),
               "check_scseed_v1_vs_posterior_maxabs": float(max(np.abs(S[f"sc{s}_v1"] - P[s]).max() for s in SEEDS)),
               "reference": {"oracle": summarise(orac, lo, hi), "best_single": summarise(best, lo, hi),
                             "mixture": summarise(mix, lo, hi)},
               "reference_posterior_scale": {"oracle": summarise(M.oracle_rates(Rpost), lo, hi),
                                             "best_single": summarise(Rpost[int(np.argmax([M.restricted_auc(Rpost[i], lo, hi) for i in range(len(SEEDS))]))], lo, hi),
                                             "mixture": summarise(M.mixture_rates(Rpost), lo, hi)},
               "by_v": {}}
        L = row["reference"]["oracle"]["rauc"] - row["reference"]["best_single"]["rauc"]
        row["L_rauc"] = L
        for v in VOTES:
            pool_v, poolb_v = S[f"scpool_v{v}"], S[f"scpoolb_v{v}"]
            single_v = {s: S[f"sc{s}_v{v}"] for s in SEEDS}
            bsv = max(single_v, key=lambda s: M.restricted_auc(single_v[s], lo, hi))
            ent = {"scpool": summarise(pool_v, lo, hi), "scpoolb": summarise(poolb_v, lo, hi),
                   "best_scseed": {"seed": bsv, **summarise(single_v[bsv], lo, hi)},
                   "scseed_of_best_single": summarise(single_v[SEEDS[best_i]], lo, hi)}
            for k in ("scpool", "scpoolb", "best_scseed", "scseed_of_best_single"):
                g = ent[k]["rauc"] - row["reference"]["best_single"]["rauc"]
                ent[k]["gain_vs_best_single"] = g
                ent[k]["frac_of_L"] = g / L if L > 1e-9 else None
            # the budget-matched comparison: pooled vote against the same seed's own vote
            ent["scpool_minus_scseed_of_best_single"] = ent["scpool"]["rauc"] - ent["scseed_of_best_single"]["rauc"]
            pb = M.paired_bootstrap_gap(pool_v, single_v[SEEDS[best_i]], lo, hi, n_boot=2000, seed=0)
            ent["samebudget_ci95"] = [float(pb.get("ci_lo", float("nan"))), float(pb.get("ci_hi", float("nan")))] if isinstance(pb, dict) else None
            pb2 = M.paired_bootstrap_gap(pool_v, best, lo, hi, n_boot=2000, seed=0)
            ent["scpool"]["gain_ci95"] = [float(pb2.get("ci_lo", float("nan"))), float(pb2.get("ci_hi", float("nan")))] if isinstance(pb2, dict) else None
            row["by_v"][str(v)] = ent
        res[split] = row
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    for split in ("val", "test"):
        r = res[split]; ref = r["reference"]
        print(f"split={split} n={r['n_prompts']} best_single={r['best_single']} "
              f"check_mix={r['check_scpool_v1_vs_mixture_of_v1_maxabs']:.4f} (MC noise ~0.03 max over 240)")
        print(f"  ref (raw scale) rAUC oracle={ref['oracle']['rauc']:.4f} best={ref['best_single']['rauc']:.4f} "
              f"mix={ref['mixture']['rauc']:.4f} L={r['L_rauc']:+.4f}")
        for v in VOTES:
            e = r["by_v"][str(v)]; ci = e.get("samebudget_ci95") or [float("nan")]*2; gci = e["scpool"].get("gain_ci95") or [float("nan")]*2
            print(f"  v={v} scpool={e['scpool']['rauc']:.4f} gain_vs_best={e['scpool']['gain_vs_best_single']:+.4f} "
                  f"[{gci[0]:+.3f},{gci[1]:+.3f}] fracL={e['scpool']['frac_of_L'] if e['scpool']['frac_of_L'] is None else round(e['scpool']['frac_of_L'],3)} | "
                  f"samebudget={e['scpool_minus_scseed_of_best_single']:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}]")
    print(f"wrote={a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
