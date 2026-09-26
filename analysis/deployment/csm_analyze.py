"""Selection, oracle, mixture and routing summaries (rAUC, Pass@1) for the seed pools."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules

import argparse
import itertools
import json
import sys

import numpy as np

import csm_metrics as M
from csm_rollouts import rate_matrix, validate, write_synthetic_cache, generated_tokens

TAUS_DEFAULT = (0.05, 0.10, 0.20, 0.30)


def _rates(cache, split, models, condition="C0", shrink=True):
    pids, succ, draws = rate_matrix(cache, split=split, models=models, condition=condition)
    r = M.posterior_rates(succ, draws) if shrink else succ / draws
    return pids, np.atleast_2d(r), succ, draws


def e1_master_bound(R, taus, seed=0, base=None):
    """E1: measure D(tau) and L(tau), and the cap that actually gates the study.

    Two pools, deliberately. D/L on the SEED-ONLY pool are the quantities Lemma 1
    is stated for. But the estimand G_multi (Def. 1) differences two BASE-INCLUSIVE
    mixtures, and D on the seed-only pool does not bound it: with identical seeds
    D is identically 0 while G_multi is positive and GROWS with M, purely because
    the pool arm carries base weight 1/(M+1) against the comparator's 1/2. Gating
    the study on the seed-only D would let a null be declared on a statistic that
    is blind to the quantity being gated, so `D_N` -- disagreement on the
    base-inclusive pool {B, P_1..P_M} -- is reported as `cap_for_g_multi` and is
    what the E1 gate must read.
    """
    out = []
    RN = R if base is None else np.vstack([np.atleast_2d(base), R])
    for t in taus:
        D, L = M.disagreement_rate(R, t), M.latent_gap(R, t)
        _, lo, hi = M.bootstrap_ci(R, lambda X: M.disagreement_rate(X, t),
                                   n_boot=800, seed=seed)
        rec = {"tau": t, "D": D, "D_ci": [lo, hi], "L": L,
               "holds": bool(L <= D + 1e-12),
               # Lemma 1 is strict whenever D > 0: summing member coverages gives
               # L <= (1 - 1/M) D. Report the sharper cap, not the loose one.
               "L_cap_sharp": (1.0 - 1.0 / R.shape[0]) * D,
               "holds_sharp": bool(L <= (1.0 - 1.0 / R.shape[0]) * D + 1e-12)}
        if base is not None:
            D_N = M.disagreement_rate(RN, t)
            _, loN, hiN = M.bootstrap_ci(RN, lambda X: M.disagreement_rate(X, t),
                                         n_boot=800, seed=seed)
            rec.update({"D_N": D_N, "D_N_ci": [loN, hiN],
                        "cap_for_g_multi": D_N})
        out.append(rec)
    return out


def e2_budget(R, tau_lo, tau_hi, draws):
    rau = M.restricted_auc(M.mixture_rates(R), tau_lo, tau_hi)
    ks = [1, 2, 8, 32, 128]
    return {"rauc_from_rates": rau,
            "rauc_is_budget_free": True,
            "pass_at_k": {k: M.pass_at_k(M.mixture_rates(R), k) for k in ks},
            "pass_at_k_increasing": bool(all(
                M.pass_at_k(M.mixture_rates(R), ks[i]) <
                M.pass_at_k(M.mixture_rates(R), ks[i + 1]) for i in range(len(ks) - 1))),
            "median_draws_per_cell": int(np.median(draws))}


def e3_base_arms(base, R, tau_lo, tau_hi):
    seed_only = M.restricted_auc(M.mixture_rates(R), tau_lo, tau_hi)
    with_base = M.restricted_auc(M.mixture_rates(np.vstack([base, R])), tau_lo, tau_hi)
    per = [M.restricted_auc(M.mixture_rates(np.vstack([base, R[m]])), tau_lo, tau_hi)
           for m in range(R.shape[0])]
    return {"base_only": M.restricted_auc(base, tau_lo, tau_hi),
            "pool_no_base": seed_only, "pool_with_base": with_base,
            "base_plus_single": per,
            "gap_survives_base_inclusion": bool(with_base > max(per) + 1e-12)}


def e4_leave_one_out(R, tau_lo, tau_hi, anchor):
    full = M.restricted_auc(M.mixture_rates(R), tau_lo, tau_hi)
    contrib = {}
    for m in range(R.shape[0]):
        if m == anchor:
            continue
        keep = [i for i in range(R.shape[0]) if i != m]
        contrib[m] = full - M.restricted_auc(M.mixture_rates(R[keep]), tau_lo, tau_hi)
    pos = sum(1 for v in contrib.values() if v > 0)
    return {"pool_rauc": full, "marginal": contrib,
            "n_positive_non_anchor": pos,
            "two_or_more_contributors": bool(pos >= 2)}


def e5_factor_m(R, taus):
    rows = []
    n = R.shape[0]
    for m in range(2, n + 1):
        for members in itertools.combinations(range(n), m):
            sub = R[list(members)]
            for t in taus:
                fl = M.mixture_floor(sub, t)
                if not np.isfinite(fl):
                    continue
                mix = M.cover_at_tau(M.mixture_rates(sub), t)
                rows.append({"M": m, "members": list(members), "tau": t,
                             "mixture": mix, "floor": fl, "slack": mix - fl,
                             "holds": bool(mix >= fl - 1e-9)})
    viol = [r for r in rows if not r["holds"]]
    slacks = [r["slack"] for r in rows]
    return {"cells": len(rows), "violations": len(viol),
            "worst_slack": min(slacks, default=float("nan")),
            "violating_cells": viol[:8],
            # NOT corroboration. Cov_mix(tau) >= Cov_orac(M tau) is arithmetic on
            # non-negative rates: it holds for EVERY input, so zero violations is
            # guaranteed and says nothing about RLVR seeds. Verified over 74,436
            # cells from 7 generators incl. adversarial ones: 0 violations always.
            "check_type": "consistency",
            "consistent": not viol,
            "note": "identity-like; a violation indicates an estimator or harness "
                    "bug, not a false theorem",
            # THE quantity that can actually come out either way: how much slack
            # real seeds leave above the floor. Small slack = specialist regime
            # (the paper's interesting case); large slack = near-duplicate regime.
            "median_slack": float(np.median(slacks)) if slacks else float("nan"),
            "frac_cells_slack_below_0.05":
                float(np.mean([s < 0.05 for s in slacks])) if slacks else float("nan")}


def e6_certificate(R, taus):
    rows, bad = [], []
    n = R.shape[0]
    for m in range(2, n + 1):
        for members in itertools.combinations(range(n), m):
            sub = R[list(members)]
            for t in taus:
                c = M.mixture_certificate(sub, t)
                if not np.isfinite(c["certificate"]):
                    continue
                row = {"M": m, "tau": t, "certificate": c["certificate"],
                       "gain": c["realised_gain"], "guaranteed": c["guaranteed"]}
                rows.append(row)
                if c["guaranteed"] and c["realised_gain"] <= 0:
                    bad.append(row)
    tab = {"cert>0,gain>0": 0, "cert>0,gain<=0": 0, "cert<=0,gain>0": 0, "cert<=0,gain<=0": 0}
    for r in rows:
        k = ("cert>0," if r["certificate"] > 0 else "cert<=0,") + \
            ("gain>0" if r["gain"] > 0 else "gain<=0")
        tab[k] += 1
    # A cell with a NON-positive certificate satisfies the implication vacuously.
    # Only positive-certificate cells put Corollary 4 at risk, so a run in which
    # the certificate never fires has TESTED NOTHING and must not read as a pass.
    fired = tab["cert>0,gain>0"] + tab["cert>0,gain<=0"]
    return {"cells": len(rows), "contingency": tab,
            "violations": len(bad), "violating_cells": bad[:8],
            "check_type": "consistency",
            "cells_certificate_fired": fired,
            "frac_cells_certificate_fired": fired / len(rows) if rows else 0.0,
            "verdict": ("untested — the certificate is positive in no cell, so "
                        "Corollary 4 was never put at risk" if fired == 0
                        else ("consistent" if not bad else "VIOLATED")),
            "consistent": not bad,
            "note": "follows from the factor-M inequality; non-positive-certificate "
                    "cells pass vacuously and must be reported separately"}


def e7_routing(R, taus, assignment=None, seed=0):
    rng = np.random.default_rng(seed)
    if assignment is None:                     # random router: the honest floor
        assignment = rng.integers(0, R.shape[0], R.shape[1])
    rows = []
    for t in taus:
        r = M.routing_loss(R, assignment, t)
        rows.append({"tau": t, **{k: (float(v) if isinstance(v, (int, float, np.floating))
                                     else v) for k, v in r.items()}})

    # IN-SAMPLE identity: loss == D * e_D is P(A n B) = P(B) P(A|B) with e_D
    # DEFINED as that conditional. It holds to machine precision for any router,
    # however bad (verified: max deviation 1.11e-16 over 18,000 cells). A paired
    # bootstrap of the difference is a point mass at zero, so "equality within
    # estimation error" is satisfied by construction and corroborates nothing.
    #
    # OUT-OF-SAMPLE version, which CAN fail: estimate e_D on half the prompts and
    # predict the coverage loss on the disjoint other half. That is a real
    # prediction -- it fails whenever the router's error rate on the disagreement
    # set does not transfer across prompts.
    n = R.shape[1]
    perm = rng.permutation(n)
    A_idx, B_idx = perm[: n // 2], perm[n // 2:]
    oos = []
    for t in taus:
        rA = M.routing_loss(R[:, A_idx], np.asarray(assignment)[A_idx], t)
        rB = M.routing_loss(R[:, B_idx], np.asarray(assignment)[B_idx], t)
        predicted = rB["D"] * rA["e_D"]          # D from B, error rate from A
        actual = rB["loss"]
        oos.append({"tau": t, "e_D_fitted_on_A": rA["e_D"], "D_on_B": rB["D"],
                    "predicted_loss_B": predicted, "actual_loss_B": actual,
                    "abs_error": abs(actual - predicted)})
    errs = [o["abs_error"] for o in oos]
    return {"cells": rows,
            "identity_holds_everywhere": bool(all(r["identity_holds"] for r in rows)),
            "check_type": "consistency",
            "consistent": bool(all(r["identity_holds"] for r in rows)),
            "note": "the in-sample identity is definitional (e_D is the conditional "
                    "it is multiplied by) and cannot fail; use out_of_sample",
            "out_of_sample": oos,
            "oos_max_abs_error": float(max(errs)) if errs else float("nan"),
            "oos_mean_abs_error": float(np.mean(errs)) if errs else float("nan")}


def g_multi(base, R, tau_lo, tau_hi, seed=0, R_val=None):
    """Conservative multi-seed gain, plus the controls that de-confound it.

    TWO controls are needed, not one, and they correct different things.

    The base-weight control (below) removes the de-dilution channel.

    The COMPARATOR-SELECTION control removes an in-sample maximum. By default
    the comparator is the best single parent measured on the SAME split it is
    then evaluated on. That was chosen deliberately, and the choice was
    conservative for the claim the study originally set out to make, namely that
    pooling seeds HELPS: an in-sample-best comparator is the hardest baseline a
    positive gain has to clear.

    That conservatism inverts once the claim inverts. The finding is now that
    pooling does NOT help, and an inflated comparator makes a null easier to
    obtain, not harder. So the same default now cuts in our favour. Passing
    `R_val` selects the comparator on the validation split instead, out of
    sample, and both are reported. Where they disagree the honest reading is the
    weaker of the two.

    G_multi differences B(+){P_1..P_M} against B(+)P_m*. The two arms carry
    DIFFERENT base weight -- 1/(M+1) versus 1/2 -- so a positive G_multi is
    attainable with zero member complementarity, just by diluting a weak base
    less. `g_multi_matched` removes that channel by replicating the BEST SEED to
    M copies alongside one base, so both arms have M+1 members and base weight
    1/(M+1), and the only remaining difference is whether the non-base mass is
    spread across the seeds or concentrated on the best one -- which is exactly
    the complementarity the paper claims. (Replicating the BASE instead would give
    it weight M/(M+1) and match nothing.) Report both: a positive G_multi with a
    null g_multi_matched is de-dilution, not complementarity, and E3 does not
    catch it because both of its arms are already base-inclusive.
    """
    g = M.g_multi(R, base, tau_lo, tau_hi)
    base = np.atleast_1d(np.asarray(base, float))
    m_star = g["comparator_index"]
    pool = M.mixture_rates(np.vstack([base, R]))
    comp = M.mixture_rates(np.vstack([base, R[m_star]]))
    pb = M.paired_bootstrap_gap(pool, comp, tau_lo, tau_hi, n_boot=1500, seed=seed)

    n_seeds = R.shape[0]
    comp_matched = M.mixture_rates(
        np.vstack([base, np.tile(R[m_star], (n_seeds, 1))]))
    matched_rauc = M.restricted_auc(comp_matched, tau_lo, tau_hi)
    pb_matched = M.paired_bootstrap_gap(pool, comp_matched, tau_lo, tau_hi,
                                        n_boot=1500, seed=seed)
    out = {**g, **pb,
           "comparator_rauc_base_matched": matched_rauc,
           "g_multi_matched": g["pool_rauc"] - matched_rauc,
           "g_multi_matched_ci": [pb_matched["ci_lo"], pb_matched["ci_hi"]],
           "g_multi_matched_lcb": pb_matched["one_sided_lcb"],
           "de_dilution_term": g["g_multi"] - (g["pool_rauc"] - matched_rauc)}

    if R_val is not None:
        mv = int(np.argmax(np.atleast_2d(R_val).mean(axis=1)))
        comp_v = M.restricted_auc(M.mixture_rates(np.vstack([base, R[mv]])),
                                  tau_lo, tau_hi)
        matched_v_rates = M.mixture_rates(
            np.vstack([base, np.tile(R[mv], (n_seeds, 1))]))
        matched_v = M.restricted_auc(matched_v_rates, tau_lo, tau_hi)
        pb_v = M.paired_bootstrap_gap(pool, matched_v_rates, tau_lo, tau_hi,
                                      n_boot=1500, seed=seed)
        out.update({
            "oos_comparator_index": mv,
            "oos_comparator_selected_on": "validation",
            "g_multi_oos": g["pool_rauc"] - comp_v,
            "g_multi_matched_oos": g["pool_rauc"] - matched_v,
            "g_multi_matched_oos_ci": [pb_v["ci_lo"], pb_v["ci_hi"]],
            "g_multi_matched_oos_lcb": pb_v["one_sided_lcb"],
            "de_dilution_term_oos": (g["pool_rauc"] - comp_v)
                                    - (g["pool_rauc"] - matched_v),
            "sign_flips_with_comparator_choice":
                bool((g["pool_rauc"] - matched_rauc) *
                     (g["pool_rauc"] - matched_v) < 0)})
    return out


def run(cache, splits_path, base, seeds, tau_lo, tau_hi, taus, out_path, seed=0):
    from csm_splits import load_splits, assert_reportable
    man = load_splits(splits_path) if splits_path else None
    info = validate(cache, man)
    print(f"cache: {info['n_records']} records, models={info['models']}")

    models = [base] + list(seeds)
    res = {"config": {"base": base, "seeds": list(seeds), "tau_lo": tau_lo,
                      "tau_hi": tau_hi, "taus": list(taus)}}

    # Validation member rates, used to select the G_multi comparator OUT OF
    # SAMPLE. Without this the comparator is the in-sample best, which inflates
    # the baseline and makes a null gain easier to obtain -- the direction that
    # now favours our own conclusion. See g_multi's docstring.
    R_val = None
    if "val" in info["splits"]:
        _, allR_val, _, _ = _rates(cache, "val", models)
        R_val = allR_val[1:]

    for split in ("val", "test"):
        if split not in info["splits"]:
            print(f"  (no '{split}' split in cache; skipping)")
            continue
        pids, allR, succ, draws = _rates(cache, split, models)
        if man is not None and split == "test":
            assert_reportable(pids, man, arm="reported test numbers")
        B, R = allR[0], allR[1:]
        print(f"\n=== split '{split}': {R.shape[0]} seeds x {R.shape[1]} prompts ===")
        anchor = int(np.argmax([M.cover_at_tau(R[m], tau_lo) for m in range(R.shape[0])]))
        blk = {
            "n_prompts": R.shape[1],
            "E1_master_bound": e1_master_bound(R, taus, seed, base=B),
            "E2_budget": e2_budget(R, tau_lo, tau_hi, draws),
            "E3_base_arms": e3_base_arms(B, R, tau_lo, tau_hi),
            "E4_leave_one_out": e4_leave_one_out(R, tau_lo, tau_hi, anchor),
            "E5_factor_M": e5_factor_m(R, taus),
            "E6_certificate": e6_certificate(R, taus),
            "E7_routing": e7_routing(R, taus, seed=seed),
            "G_multi": g_multi(B, R, tau_lo, tau_hi, seed, R_val=R_val),
            "gen_tokens": generated_tokens(cache, split=split, models=models),
        }
        res[split] = blk
        print("  [E1/E5/E6/E7 are CONSISTENCY checks: they cannot fail on any "
              "input. A VIOLATED line means a harness bug, not a refuted theorem.]")
        for t in blk["E1_master_bound"]:
            cap = t.get("cap_for_g_multi")
            print(f"  tau={t['tau']:.2f}  D={t['D']:.4f}  L={t['L']:.4f}  "
                  f"L<=(1-1/M)D {'CONSISTENT' if t['holds_sharp'] else 'VIOLATED'}"
                  + (f"  D_N(cap for G_multi)={cap:.4f}" if cap is not None else ""))
        for nm, key in (("E5 factor-M law", "E5_factor_M"),
                        ("E6 certificate", "E6_certificate"),
                        ("E7 routing identity", "E7_routing")):
            b = blk[key]
            cells = b["cells"]
            ncell = len(cells) if isinstance(cells, list) else cells
            print(f"  {nm:22s} {'CONSISTENT' if b['consistent'] else 'VIOLATED'}"
                  f"   cells={ncell} violations={b.get('violations', 0)}")
        e5, e6, e7 = blk["E5_factor_M"], blk["E6_certificate"], blk["E7_routing"]
        print("  -- informative statistics (these CAN come out either way) --")
        print(f"  E5 median slack over the floor = {e5['median_slack']:.4f}  "
              f"(frac cells < 0.05: {e5['frac_cells_slack_below_0.05']:.3f})")
        print(f"  E6 certificate fired in {e6['cells_certificate_fired']}/{e6['cells']} "
              f"cells -> {e6['verdict']}")
        print(f"  E7 OUT-OF-SAMPLE mean|actual-predicted| = {e7['oos_mean_abs_error']:.4f}  "
              f"max = {e7['oos_max_abs_error']:.4f}")
        g = blk["G_multi"]
        print(f"  G_multi={g['g_multi']:+.4f}  95% CI [{g['ci_lo']:+.4f},{g['ci_hi']:+.4f}]"
              f"  one-sided LCB={g['one_sided_lcb']:+.4f}"
              f"  positive LCB={g['positive_lcb']}")

        if "g_multi_matched_oos" in g:
            print(f"  G_multi matched: in-sample comparator "
                  f"{g['g_multi_matched']:+.4f}, out-of-sample comparator "
                  f"{g['g_multi_matched_oos']:+.4f} "
                  f"[{g['g_multi_matched_oos_ci'][0]:+.4f},"
                  f"{g['g_multi_matched_oos_ci'][1]:+.4f}]"
                  f"{'  SIGN FLIPS with comparator choice' if g['sign_flips_with_comparator_choice'] else ''}")
        print(f"  E4: {blk['E4_leave_one_out']['n_positive_non_anchor']} non-anchor seeds "
              f"contribute; two-or-more={blk['E4_leave_one_out']['two_or_more_contributors']}")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, default=float)
        print(f"\nwrote {out_path}")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cache"); ap.add_argument("--splits", default=None)
    ap.add_argument("--base", default="base")
    ap.add_argument("--seeds", nargs="+", default=["seed1", "seed2", "seed3"])
    ap.add_argument("--tau-lo", type=float, default=0.05)
    ap.add_argument("--tau-hi", type=float, default=0.30)
    ap.add_argument("--taus", nargs="+", type=float, default=list(TAUS_DEFAULT))
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        print("SELFTEST on a SYNTHETIC cache. These are not RLVR results.\n")
        from pathlib import Path
        _sd = Path(__file__).resolve().parents[2] / "analysis" / "selftest"
        _sd.mkdir(parents=True, exist_ok=True)
        write_synthetic_cache(str(_sd / "synth_cache"))
        run(str(_sd / "synth_cache"), None, "base", ["seed1", "seed2", "seed3"],
            0.05, 0.30, TAUS_DEFAULT, str(_sd / "synth_results.json"), seed=0)
        print("\nselftest complete: the pipeline runs end to end on cached counts alone.")
        return 0
    if not a.cache:
        ap.error("--cache is required unless --selftest is given")
    run(a.cache, a.splits, a.base, a.seeds, a.tau_lo, a.tau_hi, a.taus, a.out, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
