"""Matched split-half gap on every verdict set except Geometry3K test, whose values come from lib/nulls_extra.py."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import json, os
import numpy as np
import nulls_extra as ne

ROOT = str(_REPO_ROOT)
D, A = os.path.join(ROOT, "data"), os.path.join(ROOT, "analysis", "audit_data")
G, V = (0.05, 0.10, 0.20, 0.30), (0.5, 0.6, 0.7, 0.8, 0.9)
SETS = {"Geometry3K val": (D + "/geometry3k/verdicts.npz", "validation", G), "MathVista val": (D + "/mathvista/verdicts.npz", "validation", V),
        "MathVista test": (D + "/mathvista/verdicts.npz", "test", V), "Geometry3K test, k=64": (A + "/k64_geometry3k/verdicts.npz", "test", G),
        "MathVista test, k=64": (A + "/k64_mathvista/verdicts.npz", "test", V), "Geometry3K, 361 new": (A + "/ext_geometry3k/verdicts.npz", "test", G),
        "Geometry3K, 601 pooled": (A + "/pool601_geometry3k/verdicts.npz", "test", G), "3B val": (A + "/bb2_geometry3k/verdicts.npz", "validation", G),
        "3B test": (A + "/bb2_geometry3k/verdicts.npz", "test", G)}


def main():
    rng = np.random.default_rng(0); out = {}
    for name, (path, split, taus) in SETS.items():
        X, _ = ne.load_set(path, split); n, M, k = X.shape
        for t, v in ne.split_half(X, k, rng, reps=200, taus=taus).items():
            out[f"{name}|{t}"] = float(v["matched"] if isinstance(v, dict) else v)
        print(name, {t: round(out[f"{name}|{t}"], 3) for t in taus})
    json.dump(out, open(os.path.join(ROOT, "analysis", "split_half_matched.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
