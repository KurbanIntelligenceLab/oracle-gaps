"""Coverage of each cross-dataset member on each dataset's prompts, for both recipes, and the stronger pools'
router gains against the member selected on validation."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import collections, csv, glob, json, math, os
import numpy as np
import nulls_extra as ne
import specialist_pool as sp

ROOT = str(_REPO_ROOT)
TAUS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)


def light():
    cross = sp.cross_counts(); ids, C, dom, k = sp.union_counts([1], [1], "test", cross)
    print("light recipe, two members (Geometry3K-trained, MathVista-trained):")
    for t in TAUS:
        clear = C >= ne.need(t, k)
        print(f"  tau={t:.2f} Geometry3K prompts {clear[dom == 0, 0].mean():.3f} vs {clear[dom == 0, 1].mean():.3f} | MathVista prompts {clear[dom == 1, 0].mean():.3f} vs {clear[dom == 1, 1].mean():.3f}")


def stronger():
    C = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "rollout_long", "sh_*.csv"))):
        for r in csv.DictReader(open(f)):
            C[(r["model"], r["dataset"], r["split"])][r["prompt_id"]] = int(r["n_correct"])
    k = 16; two = ["geometry3k_seed1L", "mathvista_seed1L"]; four = ["geometry3k_seed1L", "geometry3k_seed2L", "mathvista_seed1L", "mathvista_seed2L"]
    need = lambda t: math.ceil(t * k - 1e-9)
    def cov(m, split, t, ds=None):
        dss = [ds] if ds else ["geometry3k", "mathvista"]
        vals = [C[(m, d, split)][p] >= need(t) for d in dss for p in C[(m, d, split)]]
        return sum(vals) / len(vals)
    print("stronger recipe, two members (Geometry3K-trained, MathVista-trained):")
    for t in TAUS:
        print(f"  tau={t:.2f} Geometry3K prompts {cov(two[0], 'test', t, 'geometry3k'):.3f} vs {cov(two[1], 'test', t, 'geometry3k'):.3f} | MathVista prompts {cov(two[0], 'test', t, 'mathvista'):.3f} vs {cov(two[1], 'test', t, 'mathvista'):.3f}")
    L = json.load(open(os.path.join(ROOT, "analysis", "specialist_pool_long.json")))["pools"]
    print("router gain against the validation-selected member:")
    for name, members in (("two_long", two), ("all_long", four)):
        for t in (0.10, 0.50, 0.70):
            sel = max(range(len(members)), key=lambda i: cov(members[i], "val", t)); R = L[name]["routing"][next(x for x in L[name]["routing"] if abs(float(x) - t) < 1e-9)]
            base = cov(members[sel], "test", t)
            gains = {lab: round(R[fam]["coverage"] - base, 3) for lab, fam in (("identity", "domain"), ("nearest-neighbor", "embedding nearest-neighbor"), ("linear", "embedding linear"))}
            print(f"  {name} tau={t:.2f} validation selects {members[sel]} | gains {gains}")


if __name__ == "__main__":
    light(); stronger()
