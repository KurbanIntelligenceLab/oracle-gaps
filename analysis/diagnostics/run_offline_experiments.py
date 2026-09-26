#!/usr/bin/env python3
"""Offline diagnostics of a pool: split-half, comparator choice, band and estimator
sensitivity, member and router spread, pool size and token budget."""

from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, json, os, sys
import numpy as np

import io_utils
import csm_analysis as A

TAUS = (0.05, 0.10, 0.20, 0.30)      # overridden by --taus
BAND = (0.05, 0.30)                  # overridden by --band
BANDS = None                         # overridden by --bands


# ----------------------------------------------------------------------
# E1  NEGATIVE CONTROL  (primary)
# ----------------------------------------------------------------------
def negative_control(rates, verdicts, routers, args):
    """How large is L-hat on a pool with no complementarity at all?

    Slice ONE member's k rollouts into M disjoint pseudo-members. Every slice is a
    draw from the same policy, so the true D and L are exactly zero and whatever
    comes back is pure estimation artifact. Because each pseudo-member then holds
    k/M draws, the real pool is subsampled to the same per-member budget before
    the two are compared: matching draws, not matching members.
    """
    V = verdicts["verdicts"]
    n, M, k = V.shape
    out = {"k": k, "M": M, "cells": []}
    for m in range(M):
        null = A.null_pool_from_one_member(V[:, m, :], TAUS, seed=args.seed + m)
        for (Mp, tau), (D0, L0, _, kk) in null.items():
            if Mp != M:
                continue
            real = A.matched_budget_real(V, M, [tau], seed=args.seed + m)[tau]
            out["cells"].append({
                "carrier_member": int(m), "tau": tau, "k_per_pseudo_member": int(kk),
                "D_null": D0, "L_null": L0, "D_real": real[0], "L_real": real[1],
                "L_excess": real[1] - L0,
            })
    if out["cells"]:
        ex = [c["L_excess"] for c in out["cells"]]
        out["L_excess_mean"] = float(np.mean(ex))
        out["L_excess_min"] = float(np.min(ex))
        out["verdict"] = ("real pool clears its null in every cell" if min(ex) > 0 else
                          "AT LEAST ONE CELL: measured L does not clear the no-complementarity null")
    return out


# ----------------------------------------------------------------------
# E2  EXCHANGEABILITY  (primary)
# ----------------------------------------------------------------------
def exchangeability(rates, verdicts, routers, args):
    """Is the realised pool exchangeable, or is one seed systematically better?

    Under exchangeability the member index carries no information, so permuting
    member labels within each prompt leaves the spread of per-member coverage
    unchanged. A spread far above that null means quality is matched only in the
    mean, which is the weaker control the paper criticises elsewhere.
    """
    R = rates["R"]
    res = {"tests": []}
    for tau in TAUS:
        t = A.exchangeability_test(R, tau, n_perm=args.n_perm, seed=args.seed)
        t["rejects_at_05"] = bool(t["p_value"] < 0.05)
        res["tests"].append(t)
    rej = [t["tau"] for t in res["tests"] if t["rejects_at_05"]]
    res["rejected_at"] = rej
    res["verdict"] = ("not rejected at any threshold: exchangeability is supported"
                      if not rej else
                      f"REJECTED at tau in {rej}: the pool is matched only in the mean")
    return res


# ----------------------------------------------------------------------
# E3  MECHANISM PARTITION  (an identity, reported for completeness)
# ----------------------------------------------------------------------
def mechanism_partition(rates, verdicts, routers, args):
    """Where do the mixture's misses fall, against a label-shuffle null?

    On a prompt that some member clears and the mixture does not,
    tau <= max_m p_m <= M * mean_m p_m < M*tau, so the maximum lies in
    [tau, M*tau) by the averaging inequality itself. The observed fraction is
    therefore 1 on every pool and the shuffled null lands at the same value:
    this is an identity, not a test, and it is reported here so that a reader
    can see that rather than take it on trust.
    """
    R = rates["R"]
    res = {"cells": []}
    for tau in TAUS:
        if R.shape[1] * tau > 1:
            res["cells"].append({"tau": tau, "skipped": "M*tau > 1, band is vacuous"})
            continue
        res["cells"].append(A.mechanism_partition(R, tau, n_perm=args.n_perm, seed=args.seed))
    live = [c for c in res["cells"] if "skipped" not in c and c.get("p_value") is not None]
    if live:
        res["verdict"] = ("concentration in [tau, M*tau) exceeds the label-shuffle null"
                          if all(c["p_value"] < 0.05 for c in live) else
                          "concentration NOT established in every live cell")
    return res


# ----------------------------------------------------------------------
# E4  SPLIT-HALF DEBIASING OF L
# ----------------------------------------------------------------------
def split_half_L(rates, verdicts, routers, args):
    """L is a plug-in maximum over noisy rates, so it inherits the winner's curse.

    Split each member's k rollouts in two. Take the argmax on one half, score the
    selected member on the other. The selection and the scoring then use disjoint
    draws, which removes the curse from the oracle arm.
    """
    V = verdicts["verdicts"]
    n, M, k = V.shape
    if k < 4:
        return {"skipped": f"k={k} too small to split"}
    rng = np.random.default_rng(args.seed)
    res = {"k": k, "cells": []}
    for tau in TAUS:
        plug, hon = [], []
        for _ in range(args.n_split_reps):
            idx = rng.permutation(k)
            a, b_ = idx[: k // 2], idx[k // 2:]
            Ra = V[:, :, a].mean(axis=2)
            Rb = V[:, :, b_].mean(axis=2)
            pick = Ra.argmax(axis=1)
            honest_rate = Rb[np.arange(n), pick]          # scored on the held-out half
            cb = A.cov_best(Rb, tau)
            hon.append(float(np.mean(honest_rate >= tau)) - cb)
            plug.append(A.cov_oracle(Rb, tau) - cb)
        res["cells"].append({
            "tau": tau,
            "L_plug_in": float(np.mean(plug)),
            "L_split_half": float(np.mean(hon)),
            "inflation": float(np.mean(plug) - np.mean(hon)),
            "reps": args.n_split_reps,
        })
    res["note"] = ("Both arms are at k/2, so compare within this table, not against "
                   "the k=16 numbers in the paper.")
    return res


# ----------------------------------------------------------------------
# E5-E12  the remaining offline items
# ----------------------------------------------------------------------
def simplex_search(rates, verdicts, routers, args):
    """Theorem 3's conclusion on the measured profiles, its only real-pool contact."""
    R = rates["R"]
    return {"cells": [
        {k: (v.tolist() if isinstance(v, np.ndarray) else v)
         for k, v in A.simplex_search(R, tau, n_dirichlet=args.n_dirichlet,
                                      seed=args.seed).items()}
        for tau in TAUS]}


def band_sensitivity(rates, verdicts, routers, args):
    """The band is pre-registered, which is not the same as the result being insensitive to it."""
    R, b = rates["R"], rates["b"]
    m_star = int(np.argmax([A.rauc(R[:, m], *BAND) for m in range(R.shape[1])]))
    out = A.band_sensitivity(R, b, m_star, bands=BANDS) if BANDS else A.band_sensitivity(R, b, m_star)
    return {"m_star": m_star,
            "bands": [{"lo": lo, "hi": hi, **v} for (lo, hi), v in out.items()]}


def estimator_sensitivity(rates, verdicts, routers, args):
    """How much of each headline number is the beta-binomial prior carrying?"""
    V = verdicts["verdicts"]
    raw = V.mean(axis=2)
    R = rates["R"]
    return {"prior_alpha": float(rates["alpha"]), "prior_beta": float(rates["beta"]),
            "prior_strength": float(rates["alpha"] + rates["beta"]),
            "shrinkage_weight_on_prior":
                float((rates["alpha"] + rates["beta"]) /
                      (rates["alpha"] + rates["beta"] + V.shape[2])),
            "cells": [{"tau": tau,
                       "D_shrunk": A.disagreement_mass(R, tau),
                       "D_raw": A.disagreement_mass(raw, tau),
                       "L_shrunk": A.latent_gap(R, tau),
                       "L_raw": A.latent_gap(raw, tau)} for tau in TAUS]}


def split_replication(rates, verdicts, routers, args):
    """Validation rollouts already exist; reporting both splits is a free generalization check."""
    out = {}
    for sp in ("validation", "test"):
        try:
            r = io_utils.load_rates(os.path.join(args.data, "rates.npz"), split=sp)
        except io_utils.DataInconsistent:
            out[sp] = {"skipped": "split absent"}
            continue
        R = r["R"]
        out[sp] = {"n_prompts": int(R.shape[0]),
                   "cells": [{"tau": tau, "D": A.disagreement_mass(R, tau),
                              "L": A.latent_gap(R, tau),
                              "mixture_gain": A.cov_mix(R, tau) - A.cov_best(R, tau)}
                             for tau in TAUS]}
    return out


def router_spread(rates, verdicts, routers, args):
    """The two families differ by ~3 prompts of 240. Is that more than initialization noise?"""
    if routers is None:
        return {"skipped": "data/routers.npz absent"}
    C = routers["choice"]
    R = rates["R"]
    cb = {tau: A.cov_best(R, tau) for tau in TAUS}
    fams = []
    for f in range(C.shape[0]):
        per_tau = []
        for tau in TAUS:
            g = [A.cov_router(R, tau, C[f, i]) - cb[tau] for i in range(C.shape[1])]
            per_tau.append({"tau": tau, "gains": [float(x) for x in g],
                            "median": float(np.median(g)),
                            "spread": float(np.max(g) - np.min(g)),
                            "e_D": float(np.mean([A.e_D(R, tau, C[f, i])
                                                  for i in range(C.shape[1])]))})
        fams.append({"router": str(routers["router_ids"][f])
                     if "router_ids" in routers else f"family_{f}",
                     "per_tau": per_tau})
    return {"families": fams,
            "note": "If the initialization spread is of the order of the family gap, "
                    "only the pooled routing null survives."}


def routing_break_even(rates, verdicts, routers, args):
    """Theorem 4's break-even at every threshold, not only tau=0.05."""
    R = rates["R"]
    rows = []
    for tau in TAUS:
        D = A.disagreement_mass(R, tau)
        L = A.latent_gap(R, tau)
        rows.append({"tau": tau, "D": D, "L": L,
                     "break_even_eD": (L / D) if D > 0 else None})
    return {"cells": rows}


def out_of_sample_comparator(rates, verdicts, routers, args):
    """Every gain subtracts a comparator selected on the evaluated split, which inflates it."""
    out = {}
    try:
        val = io_utils.load_rates(os.path.join(args.data, "rates.npz"), split="validation")
    except io_utils.DataInconsistent:
        return {"skipped": "validation split absent, cannot select out of sample"}
    R = rates["R"]
    m_star_oos = int(np.argmax([A.rauc(val["R"][:, m], *BAND) for m in range(val["R"].shape[1])]))
    for tau in TAUS:
        in_sample = A.cov_best(R, tau)
        oos = float(np.mean(R[:, m_star_oos] >= tau))
        out[str(tau)] = {
            "comparator_in_sample": in_sample,
            "comparator_out_of_sample": oos,
            "inflation": in_sample - oos,
            "mixture_gain_in_sample": A.cov_mix(R, tau) - in_sample,
            "mixture_gain_out_of_sample": A.cov_mix(R, tau) - oos,
            "oracle_gain_in_sample": A.cov_oracle(R, tau) - in_sample,
            "oracle_gain_out_of_sample": A.cov_oracle(R, tau) - oos,
        }
    return {"m_star_out_of_sample": m_star_oos, "cells": out}


def pool_size_slope(rates, verdicts, routers, args):
    """The oracle's rise with M, and the gap column that carries the paper's claim.

    pool_size_sweep returns {(M, tau): (oracle_gain, mixture_gain)}. The third
    column, their difference, is the share the mixture leaves uncollected, and it
    is the one quantity here that does not subtract the comparator, so it is free
    of the selection inflation that touches the other two.
    """
    R = rates["R"]
    sweep = A.pool_size_sweep(R, TAUS)
    rows = [{"M": int(M), "tau": float(tau),
             "oracle_gain": float(o), "mixture_gain": float(m),
             "uncollected_gap": float(o - m)}
            for (M, tau), (o, m) in sorted(sweep.items())]
    slopes = {}
    for tau in TAUS:
        cells = [r for r in rows if r["tau"] == tau]
        if len(cells) >= 2:
            slopes[str(tau)] = {
                "oracle_gain_M2_to_Mmax": cells[-1]["oracle_gain"] - cells[0]["oracle_gain"],
                "mixture_gain_M2_to_Mmax": cells[-1]["mixture_gain"] - cells[0]["mixture_gain"],
                "gap_M2_to_Mmax": cells[-1]["uncollected_gap"] - cells[0]["uncollected_gap"],
            }
    return {"cells": rows, "change_from_smallest_to_largest_pool": slopes}


def token_budget_check(rates, verdicts, routers, args):
    """Definition 1 matches rollouts. Did that hand a verbose member more compute?"""
    if "n_gen_tokens" not in verdicts:
        return {"skipped": "n_gen_tokens absent from verdicts.npz"}
    T = verdicts["n_gen_tokens"]
    per_member = T.mean(axis=(0, 2))
    return {"mean_tokens_per_member": [float(x) for x in per_member],
            "max_over_min_ratio": float(per_member.max() / per_member.min()),
            "note": "A ratio far from 1 means matched rollouts did not match compute; "
                    "say so where Definition 1 is stated."}


EXPERIMENTS = {
    "negative_control":        (negative_control,        True,  "primary"),
    "exchangeability":         (exchangeability,         False, "primary"),
    "mechanism_partition":     (mechanism_partition,     False, "identity"),
    "split_half_L":            (split_half_L,            True,  "high"),
    "out_of_sample_comparator":(out_of_sample_comparator,False, "high"),
    "simplex_search":          (simplex_search,          False, "medium"),
    "band_sensitivity":        (band_sensitivity,        False, "medium"),
    "estimator_sensitivity":   (estimator_sensitivity,   True,  "medium"),
    "split_replication":       (split_replication,       False, "medium"),
    "router_spread":           (router_spread,           False, "medium"),
    "routing_break_even":      (routing_break_even,      False, "low"),
    "pool_size_slope":         (pool_size_slope,         False, "low"),
    "token_budget_check":      (token_budget_check,      True,  "low"),
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results/offline")
    p.add_argument("--only", nargs="*", choices=sorted(EXPERIMENTS), default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--n-dirichlet", type=int, default=4000)
    p.add_argument("--n-split-reps", type=int, default=20)
    p.add_argument("--taus", type=float, nargs="+", default=None,
                   help="threshold grid (default: the Geometry3K grid)")
    p.add_argument("--band", type=float, nargs=2, default=None,
                   help="reliability band lo hi (default: 0.05 0.30)")
    p.add_argument("--bands", type=str, nargs="+", default=None,
                   help="alternative bands for band_sensitivity, as lo:hi")
    args = p.parse_args()
    global TAUS, BAND, BANDS
    if args.taus: TAUS = tuple(args.taus)
    if args.band: BAND = tuple(args.band)
    if args.bands: BANDS = tuple(tuple(float(x) for x in b.split(":")) for b in args.bands)
    print(f"taus {TAUS}  band {BAND}  bands {BANDS}")

    names = args.only or list(EXPERIMENTS)
    need_v = any(EXPERIMENTS[n][1] for n in names)
    try:
        rates, verdicts, routers = io_utils.load_all(args.data, args.split, need_verdicts=need_v)
    except (io_utils.DataMissing, io_utils.DataInconsistent) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    print(f"data {args.data}, split {args.split}")
    print(io_utils.describe(rates, verdicts, routers))
    os.makedirs(args.out, exist_ok=True)

    rows, failures = [], 0
    for name in names:
        fn, needs_v, prio = EXPERIMENTS[name]
        if needs_v and verdicts is None:
            rows.append((name, prio, "SKIPPED (needs verdicts.npz)")); continue
        print(f"\n--- {name}  [{prio}] ---")
        try:
            res = fn(rates, verdicts, routers, args)
        except Exception as e:                       # a failure is a result, not a crash
            failures += 1
            rows.append((name, prio, f"ERROR: {type(e).__name__}: {e}"))
            print(f"  ERROR: {e}")
            continue
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(res, f, indent=2, default=float)
        summary = res.get("verdict") or res.get("skipped") or res.get("note") or "written"
        rows.append((name, prio, str(summary)[:74]))
        print(f"  -> {args.out}/{name}.json")
        print(f"  {summary}")

    print("\n" + "=" * 78)
    print(f"{'experiment':26s} {'priority':10s} outcome")
    print("-" * 78)
    for n, pr, s in rows:
        print(f"{n:26s} {pr:10s} {s}")
    print("=" * 78)
    print(f"\nResults written to {args.out}/<experiment>.json.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
