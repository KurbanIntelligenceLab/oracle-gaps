#!/usr/bin/env python3
"""Upper and lower tails of both nulls and degenerate-prompt counts for every verdict set."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nulls_extra as ne   # noqa: E402

SCRIPT_VERSION = "two_sided/1.0"
ROOT = str(_REPO_ROOT)
D = os.path.join(ROOT, "data"); A = os.path.join(ROOT, "analysis", "audit_data")
SETS = [("geometry3k_val", D + "/geometry3k/verdicts.npz", "validation", (0.05, 0.10, 0.20, 0.30)), ("geometry3k_test", D + "/geometry3k/verdicts.npz", "test", (0.05, 0.10, 0.20, 0.30)),
        ("mathvista_val", D + "/mathvista/verdicts.npz", "validation", (0.5, 0.6, 0.7, 0.8, 0.9)), ("mathvista_test", D + "/mathvista/verdicts.npz", "test", (0.5, 0.6, 0.7, 0.8, 0.9)),
        ("k64_geometry3k_test", A + "/k64_geometry3k/verdicts.npz", "test", (0.05, 0.10, 0.20, 0.30)), ("k64_mathvista_test", A + "/k64_mathvista/verdicts.npz", "test", (0.5, 0.6, 0.7, 0.8, 0.9)),
        ("ext_geometry3k_test", A + "/ext_geometry3k/verdicts.npz", "test", (0.05, 0.10, 0.20, 0.30)), ("pool601_geometry3k_test", A + "/pool601_geometry3k/verdicts.npz", "test", (0.05, 0.10, 0.20, 0.30)),
        ("3b_val", A + "/bb2_geometry3k/verdicts.npz", "validation", (0.05, 0.10, 0.20, 0.30)), ("3b_test", A + "/bb2_geometry3k/verdicts.npz", "test", (0.05, 0.10, 0.20, 0.30))]
N_PERM = 2000


def main():
    task = int(os.environ.get("TASK_ID", "0")); out = os.environ.get("OUT", "two_sided.csv"); rng = np.random.default_rng(4000 + task)
    rows = []
    for name, path, split, taus in SETS[task::4]:
        V, _ = ne.load_set(path, split); n, M, k = V.shape; C = V.sum(2); S = C.sum(1); degenerate = int(((S == 0) | (S == M * k)).sum())
        obs = ne.stats(C, k, taus); rd = [ne.stats(ne.redeal_counts(V, rng), k, taus) for _ in range(N_PERM)]
        mg = [ne.stats(c, k, taus) for c in ne.margin_chain(C, k, rng, N_PERM)]
        for t in taus:
            o = obs[t]["L"]; r = np.array([x[t]["L"] for x in rd]); m = np.array([x[t]["L"] for x in mg])
            rows.append(dict(set=name, n=n, M=M, k=k, degenerate=degenerate, tau=t, L=o, redeal_mean=r.mean(), p_up=np.mean(r >= o - 1e-12), p_low=np.mean(r <= o + 1e-12),
                             margin_mean=m.mean(), pm_up=np.mean(m >= o - 1e-12), pm_low=np.mean(m <= o + 1e-12)))
        print(f"progress set={name} n={n} degenerate={degenerate} " + " ".join(f"tau={t:.2f}:p_up={x['p_up']:.3f}/p_low={x['p_low']:.3f}" for t, x in zip(taus, rows[-len(taus):])))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"RESULT task={task} rows={len(rows)} wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
