#!/usr/bin/env python3
"""Quality-only pools under logit, probit and additive links: when the margin-preserving test rejects."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, os, sys
import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nulls_extra as ne   # noqa: E402

SCRIPT_VERSION = "link_check/1.0"
ROOT = str(_REPO_ROOT)
K, TAUS, REPS, PERM = 16, (0.05, 0.10, 0.20), 30, 400
LINKS = {"logit": (lambda p: np.log(p / (1 - p)), lambda z: 1 / (1 + np.exp(-z))),
         "probit": (norm.ppf, norm.cdf),
         "additive": (lambda p: p, lambda z: np.clip(z, 0, 1))}


def solve_shift(inv, fwd, a, gap):
    """member offset b<0 such that the weak group's mean rate is lower by `gap`."""
    lo, hi = -6.0, 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if fwd(a).mean() - fwd(a + mid).mean() > gap: lo = mid
        else: hi = mid
    return (lo + hi) / 2


def main():
    task = int(os.environ.get("TASK_ID", "0")); out = os.environ.get("OUT", "link_check.csv"); rng = np.random.default_rng(3000 + task)
    V, _ = ne.load_set(os.path.join(ROOT, "data", "geometry3k", "verdicts.npz"), "test"); C = V.sum(2); n, M, k = V.shape
    base = np.clip(C.sum(1) / (M * k), 1 / (2 * M * k), 1 - 1 / (2 * M * k))
    configs = [(link, gap) for link in ("logit", "probit", "additive") for gap in (0.13, 0.20)][task::4]
    print(f"run={SCRIPT_VERSION} task={task} configs={configs} n={n} k={K}")
    rows = []
    for link, gap in configs:
        inv, fwd = LINKS[link]; a = inv(base); b = solve_shift(inv, fwd, a, gap)
        R = np.concatenate([np.repeat(fwd(a)[:, None], 5, 1), np.repeat(fwd(a + b)[:, None], 5, 1)], 1)
        for r in range(REPS):
            Cs = rng.binomial(K, R); Vs = ne.counts_to_V(Cs, K, rng)
            res = ne.both_nulls(Vs, K, rng, PERM, taus=TAUS, keys=("L",))
            for t in TAUS:
                rows.append(dict(link=link, gap=gap, rep=r, tau=t, mean_strong=float(fwd(a).mean()), mean_weak=float(fwd(a + b).mean()),
                                 L=res[t]["L"]["redeal"]["obs"], p_redeal=res[t]["L"]["redeal"]["p"], p_margin=res[t]["L"]["margin"]["p"]))
        for t in TAUS:
            rr = [x for x in rows if x["link"] == link and x["gap"] == gap and x["tau"] == t]
            print(f"progress link={link} gap={gap} tau={t:.2f} reject_redeal={np.mean([x['p_redeal'] < 0.05 for x in rr]):.2f} reject_margin={np.mean([x['p_margin'] < 0.05 for x in rr]):.2f}")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"RESULT task={task} rows={len(rows)} wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
