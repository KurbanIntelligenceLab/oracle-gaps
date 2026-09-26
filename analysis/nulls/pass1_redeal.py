"""Pass@1 oracle gap (mean per-prompt best rate minus the best member's mean rate) under both nulls."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import json, os
import numpy as np
import nulls_extra as ne

ROOT = str(_REPO_ROOT)


def pass1_stat(counts, k):
    R = counts / k
    return float(R.max(1).mean() - R.mean(0).max()), float(R.max(1).mean()), float(R.mean(0).max())


def main():
    rng = np.random.default_rng(0); out = {}
    for ds in ("geometry3k", "mathvista"):
        z = np.load(os.path.join(ROOT, "data", ds, "verdicts.npz")); out[ds] = {}
        for split in ("validation", "test"):
            m = z["split"] == split; V = z["verdicts"][m].astype(np.uint8); n, M, k = V.shape; C = V.sum(2)
            obs, orac, best = pass1_stat(C, k)
            rd = np.array([pass1_stat(ne.redeal_counts(V, rng), k)[0] for _ in range(2000)])
            mg = np.array([pass1_stat(c, k)[0] for c in ne.margin_chain(C, k, rng, 2000)])
            out[ds][split] = {"oracle": orac, "best": best, "L1": obs, "redeal_null": float(rd.mean()), "p": float(np.mean(rd >= obs - 1e-12)),
                              "margin_null": float(mg.mean()), "p_m": float(np.mean(mg >= obs - 1e-12))}
            print(f"{ds} {split}: L1={obs:.3f} re-deal {rd.mean():.3f} p={out[ds][split]['p']:.3f} | margin {mg.mean():.3f} p_m={out[ds][split]['p_m']:.3f}")
    os.makedirs(os.path.join(ROOT, "analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "analysis", "pass1_redeal.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
