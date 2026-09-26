#!/usr/bin/env python3
"""Fit the text router families on validation and assign the test prompts."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import glob, json, os, sys

DATASET = os.environ.get("CSM_DATASET", "geometry3k")
CACHE = os.environ.get("CSM_CACHE", "analysis/cache_m5pool")
# The extensions fit on one cache's validation split and score another's test split
# (routers fit on the 7B validation probe, applied to the 361 extra prompts), and
# their question text sits under raw split labels (testext, b3val, ...). Both default
# to the primary study's layout, so the original job is unchanged.
CACHE_VAL = os.environ.get("CSM_CACHE_VAL", CACHE)
TEXT_SPLITS = {"val": os.environ.get("CSM_TEXT_SPLIT_VAL", "val").split(","),
               "test": os.environ.get("CSM_TEXT_SPLIT_TEST", "test").split(",")}


def load_text(arm, split, data_root):
    """prompt_id -> question, from the persisted raw completions (one or more raw split dirs)."""
    out = {}
    for raw_split in TEXT_SPLITS[split]:
        d = os.path.join(data_root, "CSM", "rollouts_raw", f"{arm}__{DATASET}__{raw_split}")
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            for line in open(f):
                line = line.strip()
                if line:
                    r = json.loads(line)
                    out[r["prompt_id"]] = r["question"]
    return out

def text_features(texts):
    """Label-free numeric features for the gradient-boosted router family.

    The gbm_percorrect family predicts each member's rate from features and then
    takes the argmax, so it needs numbers rather than raw text. Everything here
    is computed from the QUESTION alone, never from an answer or a rate, so the
    arm stays honest: no test label touches the fit.
    """
    import re as _re
    KEYS = ("angle", "circle", "triangle", "parallel", "perpendicular",
            "radius", "diameter", "area", "perimeter", "arc", "tangent",
            "similar", "congruent", "trapezoid", "rhombus")
    rows = []
    for t in texts:
        t = str(t or "")
        nums = _re.findall(r"\d+(?:\.\d+)?", t)
        rows.append([len(t), len(t.split()), sum(c.isdigit() for c in t),
                     len(nums), float(max((float(x) for x in nums), default=0.0)),
                     t.count("\\"), t.count("$"), t.count("\\frac"),
                     t.count("="), t.count("?")]
                    + [float(k in t.lower()) for k in KEYS])
    return rows


def load_rates(cache_dir, split, models):
    import csm_rollouts, csm_metrics as M
    pids, succ, draws = csm_rollouts.rate_matrix(cache_dir, split=split, models=models)
    return pids, M.posterior_rates(succ, draws)

def main() -> int:
    out = os.environ.get("OUT", "e7_routers.json")
    data_root = os.environ.get("DATA_DIR", "")
    import numpy as np, csm_metrics as M, csm_router
    rep = {"script_version": "e7_routers/4.4", "routers": {}, "dataset": DATASET}

    # Five seeds now, from the one combined pool cache. Both splits live in the
    # same directory, so val and test can no longer drift onto different pools.
    seeds = ["seed1", "seed2", "seed3", "seed4", "seed5"]
    cache = CACHE
    vp, vr = load_rates(CACHE_VAL, "val", seeds)
    tp, tr = load_rates(cache, "test", seeds)
    rep["members"], rep["cache"], rep["cache_val"] = seeds, cache, CACHE_VAL
    rep["text_splits"] = TEXT_SPLITS
    vtext_map = load_text("base", "val", data_root)
    ttext_map = load_text("base", "test", data_root)
    vtext = [vtext_map.get(p, "") for p in vp]
    ttext = [ttext_map.get(p, "") for p in tp]
    vfeat, tfeat = text_features(vtext), text_features(ttext)
    # the base arm, needed to put routing on the same footing as the other rungs
    _, vb = load_rates(CACHE_VAL, "val", ["base"])
    _, tb = load_rates(cache, "test", ["base"])
    vbase, tbase = vb[0], tb[0]
    rep["n_val"], rep["n_test"] = len(vp), len(tp)
    rep["text_coverage"] = {"val": sum(1 for x in vtext if x) / len(vtext),
                            "test": sum(1 for x in ttext if x) / len(ttext)}

    taus = [float(x) for x in os.environ.get("CSM_TAUS", "0.05 0.10 0.20 0.30").split()]
    lo, hi = float(os.environ.get("CSM_TAU_LO", "0.05")), float(os.environ.get("CSM_TAU_HI", "0.30"))
    rep["taus"], rep["band"] = taus, [lo, hi]
    INITS = [0, 1, 2]

    def cells_for(rr):
        cells = []
        for t in taus:
            cov_r = M.cover_at_tau(rr, t)
            cov_best = max(M.cover_at_tau(tr[m], t) for m in range(tr.shape[0]))
            cov_or = M.cover_at_tau(M.oracle_rates(tr), t)
            L = cov_or - cov_best
            cells.append({"tau": t, "cov_router": cov_r, "cov_best": cov_best,
                          "cov_oracle": cov_or, "L": L, "gain": cov_r - cov_best,
                          "frac_of_L": (cov_r - cov_best) / L if L > 1e-9 else None})
        return cells

    def summary_for(rr, cells):
        return {"cells": cells,
                "mean_frac_of_L": float(np.mean([c["frac_of_L"] for c in cells
                                                 if c["frac_of_L"] is not None])),
                # the reporting skeleton tabulates rAUC over the declared band
                # and Pass@1, so emit both here rather than deriving them later
                "rauc": float(M.restricted_auc(rr, lo, hi)),
                "pass_at_1": float(np.mean(rr)),
                # The member-only figures above are not comparable with the
                # budget-matched mixture, which carries the base at weight
                # 1/(M+1). A router that simply drops a weak base would look
                # better for the de-dilution reason this paper is about, so we
                # also report the routing arm in the SAME base-inclusive form as
                # the "best single B(+)P_m*" row: B mixed with the routed member.
                "rauc_base_inclusive": float(M.restricted_auc(
                    M.mixture_rates(np.vstack([tbase, rr])), lo, hi)),
                "pass_at_1_base_inclusive": float(np.mean(
                    M.mixture_rates(np.vstack([tbase, rr]))))}

    rep["inits"] = INITS
    for kind in csm_router.ROUTERS:
        try:
            per_init = []
            for sd in INITS:
                asg = csm_router.fit_router(
                    kind, train_text=vtext, train_rates=vr, train_feats=vfeat,
                    apply_text=ttext, apply_rates=tr, apply_feats=tfeat, seed=sd)
                # in-sample application on validation, for the audit harness only
                asg_val = csm_router.fit_router(
                    kind, train_text=vtext, train_rates=vr, train_feats=vfeat,
                    apply_text=vtext, apply_rates=vr, apply_feats=vfeat, seed=sd)
                rr = M.router_rates(tr, asg)
                d = summary_for(rr, cells_for(rr))
                d["seed"] = sd
                d["assignment"] = [int(a) for a in asg]
                d["assignment_val_in_sample"] = [int(a) for a in asg_val]
                per_init.append(d)
            # per-cell median across initializations, spread kept beside it
            med = {"per_init": per_init, "n_inits": len(INITS)}
            for key in ("rauc", "pass_at_1", "rauc_base_inclusive",
                        "pass_at_1_base_inclusive", "mean_frac_of_L"):
                vals = [d[key] for d in per_init]
                med[key] = float(np.median(vals))
                med[key + "_min"], med[key + "_max"] = float(min(vals)), float(max(vals))
            cells = []
            for i, t in enumerate(taus):
                cs = [d["cells"][i] for d in per_init]
                c = dict(cs[0])
                for key in ("cov_router", "gain"):
                    vals = [x[key] for x in cs]
                    c[key] = float(np.median(vals))
                    c[key + "_min"], c[key + "_max"] = float(min(vals)), float(max(vals))
                c["frac_of_L"] = c["gain"] / c["L"] if c["L"] > 1e-9 else None
                cells.append(c)
            med["cells"] = cells
            med["assignment"] = per_init[0]["assignment"]   # seed-0 kept for compatibility
            rep["routers"][kind] = med
        except Exception as e:
            rep["routers"][kind] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}

    json.dump(rep, open(out, "w"), indent=2, default=str)
    print(f"n_val={rep['n_val']} n_test={rep['n_test']} text_cov={rep['text_coverage']}")
    print(f"{'router':16s} {'tau':>5s} {'best':>7s} {'router':>7s} {'oracle':>7s} {'L':>7s} {'gain':>8s} {'%ofL':>7s}")
    for k, v in rep["routers"].items():
        if "error" in v:
            print(f"{k:16s} ERROR {v['error']}"); continue
        print(f"{k:16s} member-only rAUC={v['rauc']:.4f} Pass@1={v['pass_at_1']:.4f} | "
              f"base-inclusive rAUC={v['rauc_base_inclusive']:.4f} "
              f"Pass@1={v['pass_at_1_base_inclusive']:.4f}")
        for c in v["cells"]:
            fl = f"{100*c['frac_of_L']:6.1f}%" if c["frac_of_L"] is not None else "    n/a"
            print(f"{k:16s} {c['tau']:5.2f} {c['cov_best']:7.4f} {c['cov_router']:7.4f} "
                  f"{c['cov_oracle']:7.4f} {c['L']:7.4f} {c['gain']:+8.4f} {fl:>7s}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
