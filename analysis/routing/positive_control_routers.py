#!/usr/bin/env python3
"""Tests and routers on the positive controls (mixed-size pool, planted predictable specialists)."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nulls_extra as ne   # noqa: E402

SCRIPT_VERSION = "positive_control_routers/1.1"
ROOT = str(_REPO_ROOT)
DATA7 = os.path.join(ROOT, "data", "geometry3k", "verdicts.npz")
DATA3 = os.environ.get("CSM_3B_VERDICTS", os.path.join(ROOT, "analysis", "audit_data", "geometry3k_3b", "verdicts.npz"))
EMB = os.path.join(ROOT, "results", "embed_prompts")
TAUS = (0.05, 0.10, 0.20, 0.30); N_NEIGH, N_PCA = 15, 64


def load(path, split):
    z = np.load(path); m = z["split"] == split
    return list(z["prompt_ids"][m]), z["verdicts"][m].astype(np.uint8)


def embeddings(split):
    for f in glob.glob(os.path.join(EMB, "sh_*.npz")):
        z = np.load(f, allow_pickle=True)
        if str(z["dataset"]) == "geometry3k" and str(z["split"]) in (split, {"validation": "val"}.get(split, split)):
            return dict(zip(z["prompt_ids"], z["embeddings"]))
    raise FileNotFoundError(split)


def fit_routers(Xtr, Rtr, Xte):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    An = A / np.linalg.norm(A, axis=1, keepdims=True); Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    nb = np.argsort(-(Bn @ An.T), axis=1)[:, :N_NEIGH]; knn = Rtr[nb].mean(1).argmax(1)
    pca = PCA(n_components=N_PCA, random_state=0).fit(A)
    clf = LogisticRegression(max_iter=5000, C=0.5).fit(pca.transform(A), Rtr.argmax(1))
    return {"embedding nearest-neighbor": knn, "embedding linear": clf.predict(pca.transform(B))}


def evaluate(Ct, k, routers, floor_member, rng, n_boot=2000):
    """Per threshold: best member on test, oracle, L, D, each router's gain over the best member and over the floor member, e_D."""
    n, M = Ct.shape; out = {}
    for t in TAUS:
        need = ne.need(t, k); clear = Ct >= need
        orac = clear.any(1).mean(); cov = clear.mean(0); best_m = int(cov.argmax()); best = cov.max()
        dis = clear.any(1) & ~clear.all(1); D = float(dis.mean()); L = float(orac - best)
        cell = {"oracle": float(orac), "best_member": float(best), "best_member_index": best_m, "floor_member_index": int(floor_member), "floor_coverage": float(cov[floor_member]),
                "L": L, "D": D, "L_over_D": (L / D if D > 0 else None), "break_even_eD": (L / D if D > 0 else None)}
        for name, asg in routers.items():
            rr = Ct[np.arange(n), asg] >= need; gain = rr.mean() - best; gain_floor = rr.mean() - cov[floor_member]
            eD = float(np.mean(~rr[dis])) if dis.any() else 0.0
            boots = [rr[ii].mean() - clear[ii, best_m].mean() for ii in (rng.integers(0, n, n) for _ in range(n_boot))]
            cell[name] = {"coverage": float(rr.mean()), "gain": float(gain), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                          "gain_over_floor": float(gain_floor), "eD": eD, "collects": bool(D > 0 and eD < L / D)}
        out[str(t)] = cell
    return out


def plant_deterministic(C, k, spec, t_moves, rng):
    counts = C.copy(); n, M = counts.shape
    for p in range(n):
        m = spec[p]; moved = 0; others = [j for j in range(M) if j != m]
        while moved < t_moves and counts[p, m] < k and any(counts[p, j] > 0 for j in others):
            j = rng.choice([j for j in others if counts[p, j] > 0]); counts[p, j] -= 1; counts[p, m] += 1; moved += 1
    return counts


def main():
    rng = np.random.default_rng(0); out = {"script_version": SCRIPT_VERSION}
    idv7, Vv7 = load(DATA7, "validation"); idt7, Vt7 = load(DATA7, "test"); k = Vv7.shape[2]
    Ev, Et = embeddings("validation"), embeddings("test")
    Xv = np.stack([Ev[p] for p in idv7]); Xt = np.stack([Et[p] for p in idt7])
    # ---------------- (1) the 7B+3B pool
    idv3, Vv3 = load(DATA3, "validation"); idt3, Vt3 = load(DATA3, "test")
    assert idv3 == idv7 and idt3 == idt7, "3B probes must match the 7B probes prompt for prompt"
    Cv = np.concatenate([Vv7.sum(2), Vv3.sum(2)], 1); Ct = np.concatenate([Vt7.sum(2), Vt3.sum(2)], 1); M = Cv.shape[1]
    floor = int((Cv[:, :5] / k).mean(0).argmax())                     # the 7B member selected on validation ("route by size")
    routers = fit_routers(Xv, Cv / k, Xt); routers["route by size (validation-best 7B)"] = np.full(len(idt7), floor)
    res = evaluate(Ct, k, routers, floor, rng)
    nulls = ne.both_nulls(np.concatenate([Vt7, Vt3], 1), k, rng, 2000, taus=TAUS, keys=("L",))
    for t in TAUS: res[str(t)]["nulls"] = nulls[t]["L"]
    out["pool_7b_3b"] = {"n": int(len(idt7)), "M": int(M), "members": "7B seeds 1-5 then 3B seeds 1-5", "router_fit": "validation", "results": res}
    for t in TAUS:
        c = res[str(t)]; nl = nulls[t]["L"]
        line = f"7B+3B tau={t:.2f} L={c['L']:.3f} D={c['D']:.3f} p_m={nl['margin']['p']:.3f} best7B(test)={c['best_member']:.3f} floor(val)={c['floor_coverage']:.3f} break_even_eD={c['break_even_eD']:.2f}"
        for name in routers: line += f" | {name[:14]} gain={c[name]['gain']:+.3f} [{c[name]['lo']:+.3f},{c[name]['hi']:+.3f}] eD={c[name]['eD']:.2f}"
        print(line)
    # ---------------- (2) the learnable planted pool (five 7B seeds)
    from sklearn.decomposition import PCA
    mu, sd = Xv.mean(0), Xv.std(0) + 1e-6; pca1 = PCA(n_components=1, random_state=0).fit((Xv - mu) / sd)
    sv = pca1.transform((Xv - mu) / sd)[:, 0]; st = pca1.transform((Xt - mu) / sd)[:, 0]
    edges = np.quantile(sv, [0.2, 0.4, 0.6, 0.8]); specv = np.digitize(sv, edges); spect = np.digitize(st, edges)
    out["planted"] = {}
    for delta in (0.0625, 0.125):
        t_moves = int(round(delta * k))
        Cpv = plant_deterministic(Vv7.sum(2), k, specv, t_moves, rng); Cpt = plant_deterministic(Vt7.sum(2), k, spect, t_moves, rng)
        floor = int((Cpv / k).mean(0).argmax())
        routers = fit_routers(Xv, Cpv / k, Xt); routers["oracle specialist (feature known)"] = spect.copy()
        res = evaluate(Cpt, k, routers, floor, rng)
        nulls = ne.both_nulls(ne.counts_to_V(Cpt, k, rng), k, rng, 2000, taus=TAUS, keys=("L",))
        for t in TAUS: res[str(t)]["nulls"] = nulls[t]["L"]
        out["planted"][str(t_moves)] = {"moved_per_prompt": t_moves, "specialist_rule": "quintile of PCA-1 of the validation-fitted embedding", "router_fit": "planted validation", "results": res}
        for t in TAUS:
            c = res[str(t)]; nl = nulls[t]["L"]
            line = f"planted moved={t_moves} tau={t:.2f} L={c['L']:.3f} D={c['D']:.3f} p={nl['redeal']['p']:.3f} p_m={nl['margin']['p']:.3f} best(test)={c['best_member']:.3f} break_even_eD={c['break_even_eD']:.2f}"
            for name in routers: line += f" | {name[:14]} gain={c[name]['gain']:+.3f} [{c[name]['lo']:+.3f},{c[name]['hi']:+.3f}] eD={c[name]['eD']:.2f}"
            print(line)
    json.dump(out, open(os.path.join(ROOT, "analysis", "positive_control_routers.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
