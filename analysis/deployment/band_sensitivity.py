"""Budget-matched mixture against the test-selected and validation-selected members on four reliability bands."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import json, os
import numpy as np
import nulls_extra as ne

ROOT = str(_REPO_ROOT)
KEEP = ("mix_equalized", "gain_vs_in_sample", "ci_in_sample", "gain_vs_selected", "ci_selected", "best_in_sample", "best_selected")


def main():
    D = os.path.join(ROOT, "data", "geometry3k")
    V, _ = ne.load_set(D + "/verdicts.npz", "test"); Vv, _ = ne.load_set(D + "/verdicts.npz", "validation"); k = V.shape[2]
    z = np.load(D + "/rates.npz"); i = list(z["prior_split_names"]).index("test"); a, b = float(z["prior_alpha"][i]), float(z["prior_beta"][i])
    out = {}
    for band in ((0.05, 0.30), (0.05, 0.50), (0.10, 0.30), (0.05, 0.20)):
        m = ne.equalized_mixture(V, k, np.random.default_rng(0), a, b, band, V_sel=Vv)
        out[f"{band[0]:.2f}-{band[1]:.2f}"] = {key: (round(v, 4) if isinstance(v, float) else v) for key, v in m.items() if key in KEEP}
        print(band, out[f"{band[0]:.2f}-{band[1]:.2f}"])
    json.dump(out, open(os.path.join(ROOT, "analysis", "band_sensitivity_equalized.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
