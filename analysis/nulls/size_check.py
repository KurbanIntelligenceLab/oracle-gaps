#!/usr/bin/env python3
"""False-positive rates of both tests on data generated under each null."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nulls_extra as ne   # noqa: E402

SCRIPT_VERSION = "size_check/1.0"
ROOT = str(_REPO_ROOT)
REPS = int(os.environ.get("CSM_SIZE_REPS", 50)); N_PERM = int(os.environ.get("CSM_SIZE_NPERM", 400))


def main():
    task = int(os.environ.get("TASK_ID", "0")); out = os.environ.get("OUT", "size_check.csv")
    rng = np.random.default_rng(1000 + task)
    V, _ = ne.load_set(os.path.join(ROOT, "data", "geometry3k", "verdicts.npz"), "test"); n, M, k = V.shape
    counts = V.sum(2)
    print(f"run={SCRIPT_VERSION} task={task} n={n} M={M} k={k} reps={REPS} n_perm={N_PERM}")
    rows = []
    margin_sets = ne.margin_chain(counts, k, rng, REPS)          # REPS thinned draws of the conditioned law
    for gen in ("redeal", "margin"):
        rej = {"redeal_010": 0, "margin_010": 0}
        for r in range(REPS):
            c = ne.redeal_counts(V, rng) if gen == "redeal" else margin_sets[r]
            Vn = ne.counts_to_V(c, k, rng)
            res = ne.both_nulls(Vn, k, rng, N_PERM, taus=(0.05, 0.10), keys=("L",))
            row = dict(generator=gen, rep=r, p_redeal_005=res[0.05]["L"]["redeal"]["p"], p_redeal_010=res[0.10]["L"]["redeal"]["p"],
                       p_margin_005=res[0.05]["L"]["margin"]["p"], p_margin_010=res[0.10]["L"]["margin"]["p"])
            rows.append(row); rej["redeal_010"] += row["p_redeal_010"] < 0.05; rej["margin_010"] += row["p_margin_010"] < 0.05
            if (r + 1) % 10 == 0: print(f"progress gen={gen} {r+1}/{REPS} reject_redeal={rej['redeal_010']/(r+1):.2f} reject_margin={rej['margin_010']/(r+1):.2f}")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    for gen in ("redeal", "margin"):
        rr = [x for x in rows if x["generator"] == gen]
        print(f"RESULT generator={gen} reps={len(rr)} size_redeal_010={np.mean([x['p_redeal_010'] < 0.05 for x in rr]):.3f} size_margin_010={np.mean([x['p_margin_010'] < 0.05 for x in rr]):.3f} "
              f"size_redeal_005={np.mean([x['p_redeal_005'] < 0.05 for x in rr]):.3f} size_margin_005={np.mean([x['p_margin_005'] < 0.05 for x in rr]):.3f}")
    print(f"wrote={out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
