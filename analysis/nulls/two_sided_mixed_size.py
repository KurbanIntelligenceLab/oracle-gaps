"""Both tails of both nulls and the matched split-half for the pool of five 7B and five 3B seeds."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import json, os
import numpy as np
import nulls_extra as ne

ROOT = str(_REPO_ROOT)


def main():
    V7, _ = ne.load_set(os.path.join(ROOT, "data", "geometry3k", "verdicts.npz"), "test")
    V3, _ = ne.load_set(os.path.join(ROOT, "analysis", "audit_data", "bb2_geometry3k", "verdicts.npz"), "test")
    V = np.concatenate([V7, V3], 1); n, M, k = V.shape; C = V.sum(2); S = C.sum(1); deg = int(((S == 0) | (S == M * k)).sum())
    rng = np.random.default_rng(4100); taus = (0.05, 0.10, 0.20, 0.30)
    obs = ne.stats(C, k, taus); rd = [ne.stats(ne.redeal_counts(V, rng), k, taus) for _ in range(2000)]
    mg = [ne.stats(c, k, taus) for c in ne.margin_chain(C, k, rng, 2000)]
    sh = ne.split_half(V, k, rng, reps=200, taus=taus); rows = []
    for t in taus:
        o = obs[t]["L"]; r = np.array([x[t]["L"] for x in rd]); m = np.array([x[t]["L"] for x in mg])
        rows.append(dict(set="7B+3B test", n=n, M=M, k=k, degenerate=deg, tau=t, L=o, redeal_mean=r.mean(), p_up=float(np.mean(r >= o - 1e-12)),
                         p_low=float(np.mean(r <= o + 1e-12)), margin_mean=m.mean(), pm_up=float(np.mean(m >= o - 1e-12)),
                         pm_low=float(np.mean(m <= o + 1e-12)), split_half_matched=float(sh[t]["matched"])))
        print({key: (round(v, 3) if isinstance(v, float) else v) for key, v in rows[-1].items()})
    json.dump(rows, open(os.path.join(ROOT, "analysis", "two_sided_7b3b.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
