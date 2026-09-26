#!/usr/bin/env python3
"""Simulations: excess versus a known population gap, and power against diffuse differences."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nulls_extra as ne   # noqa: E402

SCRIPT_VERSION = "null_sims/1.2"
ROOT = str(_REPO_ROOT)
TAU, K, M, N = 0.10, int(os.environ.get("CSM_SIM_K", 16)), 5, 240
REPS_B, PERM_B, REPS_C, PERM_C = 30, 300, 20, 400


def pop_L(R, tau):
    return float((R.max(1) >= tau - 1e-12).mean() - (R >= tau - 1e-12).mean(0).max())


def main():
    task = int(os.environ.get("TASK_ID", "0")); out = os.environ.get("OUT", "null_sims.csv"); rng = np.random.default_rng(2000 + task)
    z = np.load(os.path.join(ROOT, "data", "geometry3k", "rates.npz")); i = list(z["prior_split_names"]).index("test")
    alpha, beta = float(z["prior_alpha"][i]), float(z["prior_beta"][i])
    V, _ = ne.load_set(os.path.join(ROOT, "data", "geometry3k", "verdicts.npz"), "test"); C = V.sum(2)
    k_data = V.shape[2]                                          # the observed budget; K is the simulated one
    print(f"run={SCRIPT_VERSION} task={task} prior=Beta({alpha:.3f},{beta:.3f}) n={N} M={M} k={K}")
    rows = []
    if task in (0, 1):
        settings = [(f, s) for f in (0.0, 0.1, 0.3, 0.5) for s in (0.05, 0.10, 0.20) if not (f == 0.0 and s != 0.05)]
        for f, s in settings[task::2]:
            for r in range(REPS_B):
                p = rng.beta(alpha, beta, N); R = np.repeat(p[:, None], M, 1)
                spec = rng.integers(0, M, N); on = rng.random(N) < f
                R[np.arange(N)[on], spec[on]] = np.minimum(1.0, p[on] + s)
                Cs = rng.binomial(K, R); Vs = ne.counts_to_V(Cs, K, rng)
                plug = ne.stats(Cs, K, (TAU,))[TAU]["L"]; null = np.mean([ne.stats(ne.redeal_counts(Vs, rng), K, (TAU,))[TAU]["L"] for _ in range(PERM_B)])
                rows.append(dict(kind="W3b", f=f, s=s, sigma="", rep=r, pop_L=pop_L(R, TAU), plug_L=plug, redeal_mean=null, excess=plug - null, p_redeal="", p_margin=""))
            sub = [x for x in rows if x["f"] == f and x["s"] == s]
            print(f"progress W3b f={f} s={s} pop_L={np.mean([x['pop_L'] for x in sub]):.3f} excess={np.mean([x['excess'] for x in sub]):+.3f}")
    else:
        sigmas = [0.0, 0.25, 0.5, 1.0][task - 2::2]
        base = C.sum(1) / (M * k_data); base = np.clip(base, 1 / (2 * M * k_data), 1 - 1 / (2 * M * k_data)); logit = np.log(base / (1 - base))
        for sigma in sigmas:
            for r in range(REPS_C):
                R = 1 / (1 + np.exp(-(logit[:, None] + rng.normal(0, sigma, (N, M)))))
                Cs = rng.binomial(K, R); Vs = ne.counts_to_V(Cs, K, rng)
                res = ne.both_nulls(Vs, K, rng, PERM_C, taus=(TAU,), keys=("L",))[TAU]["L"]
                rows.append(dict(kind="W3c", f="", s="", sigma=sigma, rep=r, pop_L=pop_L(R, TAU), plug_L=res["redeal"]["obs"], redeal_mean=res["redeal"]["null_mean"],
                                 excess=res["redeal"]["excess"], p_redeal=res["redeal"]["p"], p_margin=res["margin"]["p"]))
            sub = [x for x in rows if x["sigma"] == sigma]
            print(f"progress W3c sigma={sigma} pop_L={np.mean([x['pop_L'] for x in sub]):.3f} power_redeal={np.mean([x['p_redeal'] < 0.05 for x in sub]):.2f} power_margin={np.mean([x['p_margin'] < 0.05 for x in sub]):.2f}")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"RESULT task={task} rows={len(rows)} wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
