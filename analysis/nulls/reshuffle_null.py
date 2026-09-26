#!/usr/bin/env python3
"""Re-deal null for the oracle gap, disagreement and pool-size growth at the full response budget."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import io_utils

SCRIPT_VERSION = "reshuffle_null/1.1"


def stats(counts, k, taus):
    """counts: (n, M) successes. Returns dict tau -> (D, L, gap_orac_mix, cov_best, cov_orac, cov_mix)."""
    n, M = counts.shape
    out = {}
    for t in taus:
        need = int(np.ceil(t * k - 1e-9))
        clear = counts >= need                               # (n, M)
        orac = clear.any(1).mean()
        best = clear.mean(0).max()
        mix = (counts.sum(1) / M >= need - 1e-9).mean()      # mean rate >= tau  <=> sum >= M*need... careful
        mix = ((counts.sum(1) / (M * k)) >= t - 1e-12).mean()
        D = (clear.any(1) & ~clear.all(1)).mean()
        out[t] = (D, orac - best, orac - mix, best, orac, mix)
    return out


def L_by_size(counts, k, taus, sizes=(2, 3, 4, 5)):
    """Mean over all subsets of each size of (L, oracle-minus-mixture gap)."""
    n, M = counts.shape
    res = {}
    for s in sizes:
        vals = {t: [] for t in taus}
        for sub in itertools.combinations(range(M), s):
            st = stats(counts[:, list(sub)], k, taus)
            for t in taus: vals[t].append((st[t][1], st[t][2]))
        res[s] = {t: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) for t, v in vals.items()}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30])
    ap.add_argument("--n-perm", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v = io_utils.load_verdicts(os.path.join(a.data, "verdicts.npz"), split=a.split)
    V = v["verdicts"].astype(np.int64)                         # (n, M, k)
    n, M, k = V.shape
    obs = stats(V.sum(2), k, a.taus)
    obs_rise = L_by_size(V.sum(2), k, a.taus)
    rng = np.random.default_rng(a.seed)
    flat = V.reshape(n, M * k)
    null = {t: [] for t in a.taus}; null_rise = {t: [] for t in a.taus}
    for b in range(a.n_perm):
        idx = np.argsort(rng.random((n, M * k)), axis=1)      # independent permutation per prompt
        P = np.take_along_axis(flat, idx, axis=1).reshape(n, M, k).sum(2)
        st = stats(P, k, a.taus)
        for t in a.taus: null[t].append(st[t])
        if b < 400:                                            # the subset sweep is 26 stats per draw
            r = L_by_size(P, k, a.taus)
            for t in a.taus: null_rise[t].append((r[5][t][0] - r[2][t][0], [r[m][t][0] for m in (2, 3, 4, 5)], [r[m][t][1] for m in (2, 3, 4, 5)]))
    res = {"script_version": SCRIPT_VERSION, "n": n, "M": M, "k": k, "n_perm": a.n_perm, "cells": []}
    for t in a.taus:
        arr = np.array(null[t])                                # (n_perm, 6)
        D, L, G = obs[t][0], obs[t][1], obs[t][2]
        rise = obs_rise[5][t][0] - obs_rise[2][t][0]; nr = np.array([x[0] for x in null_rise[t]])
        nL = np.array([x[1] for x in null_rise[t]]); nG = np.array([x[2] for x in null_rise[t]])
        res["cells"].append({
            "tau": t, "D_obs": float(D), "L_obs": float(L), "gap_obs": float(G),
            "cov_best_obs": float(obs[t][3]), "cov_orac_obs": float(obs[t][4]), "cov_mix_obs": float(obs[t][5]),
            "D_null_mean": float(arr[:, 0].mean()), "D_null_p95": float(np.quantile(arr[:, 0], .95)),
            "L_null_mean": float(arr[:, 1].mean()), "L_null_p95": float(np.quantile(arr[:, 1], .95)),
            "L_null_p05": float(np.quantile(arr[:, 1], .05)),
            "p_L_ge_obs": float((arr[:, 1] >= L - 1e-12).mean()),
            "p_D_ge_obs": float((arr[:, 0] >= D - 1e-12).mean()),
            "gap_null_mean": float(arr[:, 2].mean()), "p_gap_ge_obs": float((arr[:, 2] >= G - 1e-12).mean()),
            "cov_orac_null_mean": float(arr[:, 4].mean()), "cov_best_null_mean": float(arr[:, 3].mean()),
            "L_excess_over_null": float(L - arr[:, 1].mean()),
            "rise_L_M2_to_M5_obs": float(rise), "rise_null_mean": float(nr.mean()),
            "p_rise_ge_obs": float((nr >= rise - 1e-12).mean()),
            "L_by_M_obs": [obs_rise[m][t][0] for m in (2, 3, 4, 5)],
            "gap_by_M_obs": [obs_rise[m][t][1] for m in (2, 3, 4, 5)],
            "L_by_M_null_mean": [float(x) for x in nL.mean(0)],
            "L_by_M_null_p95": [float(x) for x in np.quantile(nL, .95, axis=0)],
            "gap_by_M_null_mean": [float(x) for x in nG.mean(0)],
            "gap_by_M_null_p05": [float(x) for x in np.quantile(nG, .05, axis=0)],
            "gap_by_M_null_p95": [float(x) for x in np.quantile(nG, .95, axis=0)]})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"run={SCRIPT_VERSION} data={a.data} split={a.split} n={n} M={M} k={k} n_perm={a.n_perm}")
    for c in res["cells"]:
        print(f"  tau={c['tau']:.2f} D obs={c['D_obs']:.3f} null={c['D_null_mean']:.3f} p={c['p_D_ge_obs']:.3f} | "
              f"L obs={c['L_obs']:.3f} null={c['L_null_mean']:.3f} [p05 {c['L_null_p05']:.3f}, p95 {c['L_null_p95']:.3f}] p={c['p_L_ge_obs']:.3f} excess={c['L_excess_over_null']:+.3f} | "
              f"gap obs={c['gap_obs']:.3f} null={c['gap_null_mean']:.3f} p={c['p_gap_ge_obs']:.3f} | "
              f"riseL obs={c['rise_L_M2_to_M5_obs']:+.3f} null={c['rise_null_mean']:+.3f} p={c['p_rise_ge_obs']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
