"""Budget-matched mixture on the 3B validation split (the other replication sets come from lib/nulls_extra.py)."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import json, os
import numpy as np
import nulls_extra as ne

ROOT = str(_REPO_ROOT)


def main():
    D = os.path.join(ROOT, "analysis", "audit_data", "bb2_geometry3k")
    V, _ = ne.load_set(D + "/verdicts.npz", "validation"); n, M, k = V.shape
    z = np.load(D + "/rates.npz"); names = list(z["prior_split_names"]) if "prior_split_names" in z.files else []
    if names and "validation" in names:
        i = names.index("validation"); alpha, beta = float(z["prior_alpha"][i]), float(z["prior_beta"][i])
    else:
        alpha, beta = float(z["alpha"]), float(z["beta"])
    m = ne.equalized_mixture(V, k, np.random.default_rng(0), alpha, beta, (0.05, 0.30))
    print(f"3B validation: mixture {m['mix_equalized']:.4f} gain {m['gain_vs_in_sample']:+.4f} {m['ci_in_sample']}")
    json.dump({"split": "validation", "band": [0.05, 0.30], "alpha": alpha, "beta": beta, **m},
              open(os.path.join(ROOT, "analysis", "bb2_validation_mixture.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
