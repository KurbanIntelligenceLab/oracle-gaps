#!/usr/bin/env python3
"""Routers over image-aware prompt embeddings (linear and nearest-neighbor)."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import glob, json, os, sys
import numpy as np

SCRIPT_VERSION = "embed_routers/1.0"
ROOT = str(_REPO_ROOT)
TAUS = {"geometry3k": (0.05, 0.10, 0.20, 0.30), "mathvista": (0.5, 0.6, 0.7, 0.8, 0.9)}
SPLIT = {"val": "validation", "test": "test"}


def load_embeddings():
    E = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "embed_prompts", "sh_*.npz"))):
        z = np.load(f, allow_pickle=True)
        E[(str(z["dataset"]), SPLIT[str(z["split"])])] = (list(z["prompt_ids"]), z["embeddings"].astype(np.float64))
    return E


def rates_for(ds, split):
    z = np.load(os.path.join(ROOT, "data", ds, "rates.npz"))
    m = z["split"] == split
    return list(z["prompt_ids"][m]), z["R"][m]


def align(ids_emb, X, ids_rates):
    pos = {p: i for i, p in enumerate(ids_emb)}
    return X[[pos[p] for p in ids_rates]]


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def fit_linear(Xtr, ytr, Xte, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(64, Xtr.shape[0] - 1), random_state=seed).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)
    clf = LogisticRegression(max_iter=5000, C=0.5, random_state=seed).fit(Ztr, ytr)
    return clf.predict(Zte)


def fit_knn(Xtr, Rtr, Xte, k=15):
    A = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True); B = Xte / np.linalg.norm(Xte, axis=1, keepdims=True)
    S = B @ A.T
    nb = np.argsort(-S, axis=1)[:, :k]
    return Rtr[nb].mean(1).argmax(1)


def cover(r, t): return float(np.mean(r >= t))


def main():
    E = load_embeddings(); rng = np.random.default_rng(0)
    out = {"script_version": SCRIPT_VERSION, "families": ["embedding linear", "embedding nearest-neighbor"], "cells": []}
    for ds, taus in TAUS.items():
        if (ds, "validation") not in E or (ds, "test") not in E:
            print(f"{ds}: embeddings missing"); continue
        vids, Rv = rates_for(ds, "validation"); tids, Rt = rates_for(ds, "test")
        Xv = align(*E[(ds, "validation")], vids); Xt = align(*E[(ds, "test")], tids)
        Xv, Xt = standardize(Xv, Xt)
        n, M = Rt.shape
        yv = Rv.argmax(1)
        fams = {"embedding linear": [fit_linear(Xv, yv, Xt, sd) for sd in (0, 1, 2)],
                "embedding nearest-neighbor": [fit_knn(Xv, Rv, Xt)]}
        for fam, asgs in fams.items():
            for t in taus:
                cov_best = max(cover(Rt[:, m], t) for m in range(M)); m_val = int(np.argmax([cover(Rv[:, m], t) for m in range(M)]))
                orac = cover(Rt.max(1), t); L = orac - cov_best
                D = float(np.mean((Rt.max(1) >= t) & (Rt.min(1) < t)))
                gains = [cover(Rt[np.arange(n), a], t) - cov_best for a in asgs]
                j = int(np.argsort(gains)[len(gains) // 2]); a = asgs[j]; rr = Rt[np.arange(n), a]
                dis = (Rt.max(1) >= t) & (Rt.min(1) < t); eD = float(np.mean(rr[dis] < t)) if dis.any() else 0.0
                best_m = int(np.argmax([cover(Rt[:, m], t) for m in range(M)]))
                boots = []
                for _ in range(2000):
                    ii = rng.integers(0, n, n); boots.append(cover(rr[ii], t) - cover(Rt[ii, best_m], t))
                out["cells"].append(dict(dataset=ds, family=fam, tau=t, gain=gains[j], lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                                         gain_oos=cover(rr, t) - cover(Rt[:, m_val], t), eD=eD, L_over_D=(L / D if D > 0 else float("nan")),
                                         spread=float(max(gains) - min(gains)), n_inits=len(asgs), dim=int(Xt.shape[1])))
                c = out["cells"][-1]
                print(f"{ds:10s} {fam:28s} tau={t:.2f} gain={c['gain']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}] gain_val={c['gain_oos']:+.3f} eD={c['eD']:.3f} L/D={c['L_over_D']:.3f}")
    json.dump(out, open(os.path.join(ROOT, "analysis", "embed_routers.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
