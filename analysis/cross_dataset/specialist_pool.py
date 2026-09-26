#!/usr/bin/env python3
"""Cross-dataset pools under light training: nulls, coverage and routers on the union test set."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, csv, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nulls_extra as ne   # noqa: E402

SCRIPT_VERSION = "specialist_pool/1.2"
ROOT = str(_REPO_ROOT)
PATHS = {"data": os.path.join(ROOT, "data"), "cross": os.path.join(ROOT, "results", "rollout_cross"),
         "embed": os.path.join(ROOT, "results", "embed_prompts"), "out": os.path.join(ROOT, "analysis", "specialist_pool.json")}
TAUS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70)
SPLIT = {"val": "validation", "validation": "validation", "test": "test"}
N_FOLDS, N_NEIGH, N_PCA = 5, 15, 64


def in_domain(ds, split):
    z = np.load(os.path.join(PATHS["data"], ds, "verdicts.npz"))
    m = z["split"] == split
    return list(z["prompt_ids"][m]), z["verdicts"][m].astype(np.uint8)          # (n, 5, 16)


def cross_counts():
    """(adapter_ds, seed, eval_ds, split) -> {prompt_id: (n_correct, n_draws)}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(PATHS["cross"], "sh_*.csv"))):
        for r in csv.DictReader(open(f)):
            ads, seed = r["model"].split("_")
            out.setdefault((ads, seed, r["dataset"], SPLIT[r["split"]]), {})[r["prompt_id"]] = (int(r["n_correct"]), int(r["n_draws"]))
    return out


def complete(cross, ads, seed, eval_ds, split, ids):
    d = cross.get((ads, f"seed{seed}", eval_ds, split), {})
    return all(p in d for p in ids)


def counts_to_V(c, k, rng):
    n, M = c.shape; V = np.zeros((n, M, k), dtype=np.uint8)
    for p in range(n):
        for m in range(M):
            V[p, m, :c[p, m]] = 1; V[p, m] = V[p, m][rng.permutation(k)]
    return V


def union_counts(seeds_g, seeds_m, split, cross):
    """Union of the two probes; members = geometry3k seeds then mathvista seeds. Returns ids, counts (n, M), domain, k."""
    idg, Vg = in_domain("geometry3k", split); idm, Vm = in_domain("mathvista", split)
    k = Vg.shape[2]
    rows, doms = [], []
    for p_i, p in enumerate(idg):
        rows.append([int(Vg[p_i, s - 1].sum()) for s in seeds_g] + [cross[("mathvista", f"seed{s}", "geometry3k", split)][p][0] for s in seeds_m]); doms.append(0)
    for p_i, p in enumerate(idm):
        rows.append([cross[("geometry3k", f"seed{s}", "mathvista", split)][p][0] for s in seeds_g] + [int(Vm[p_i, s - 1].sum()) for s in seeds_m]); doms.append(1)
    return list(idg) + list(idm), np.array(rows), np.array(doms), k


def embeddings():
    E = {}
    for f in glob.glob(os.path.join(PATHS["embed"], "sh_*.npz")):
        z = np.load(f, allow_pickle=True); E[(str(z["dataset"]), SPLIT[str(z["split"])])] = dict(zip(z["prompt_ids"], z["embeddings"]))
    return E


def stack_embeddings(E, ids, dom, split):
    return np.stack([E[("geometry3k" if dom[i] == 0 else "mathvista", split)][p] for i, p in enumerate(ids)])


def fit_routers(Xtr, Rtr, Xte):
    """Nearest-neighbor and linear routers over standardized embeddings: assignments for Xte."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    An = A / np.linalg.norm(A, axis=1, keepdims=True); Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    nb = np.argsort(-(Bn @ An.T), axis=1)[:, :N_NEIGH]; asg_knn = Rtr[nb].mean(1).argmax(1)
    pca = PCA(n_components=min(N_PCA, A.shape[0] - 1), random_state=0).fit(A)
    clf = LogisticRegression(max_iter=5000, C=0.5).fit(pca.transform(A), Rtr.argmax(1))
    return asg_knn, clf.predict(pca.transform(B))


def main():
    ap = argparse.ArgumentParser()
    for key in PATHS: ap.add_argument("--" + key, default=PATHS[key])
    a = ap.parse_args(); PATHS.update(vars(a))
    rng = np.random.default_rng(0); cross = cross_counts(); E = embeddings()
    idg_t, _ = in_domain("geometry3k", "test"); idm_t, _ = in_domain("mathvista", "test")
    out = {"script_version": SCRIPT_VERSION, "pools": {}}
    for name, sg, sm in (("two_specialists", [1], [1]), ("all_seeds", [1, 2, 3, 4, 5], [1, 2, 3, 4, 5])):
        dropped = [f"geometry3k seed {s}" for s in sg if not complete(cross, "geometry3k", s, "mathvista", "test", idm_t)] + \
                  [f"mathvista seed {s}" for s in sm if not complete(cross, "mathvista", s, "geometry3k", "test", idg_t)]
        sg = [s for s in sg if f"geometry3k seed {s}" not in dropped]; sm = [s for s in sm if f"mathvista seed {s}" not in dropped]
        if len(sg) + len(sm) < 2:
            out["pools"][name] = {"skipped": "fewer than two members with complete cross scores", "members_dropped": dropped}; continue
        ids, C, dom, k = union_counts(sg, sm, "test", cross); n, M = C.shape
        V = counts_to_V(C, k, rng); R = C / k
        res = {"n": int(n), "M": int(M), "k": int(k), "members": [f"geometry3k seed {s}" for s in sg] + [f"mathvista seed {s}" for s in sm], "members_dropped": dropped,
               "domain_rates": {"geometry3k_prompts": R[dom == 0].mean(0).round(3).tolist(), "mathvista_prompts": R[dom == 1].mean(0).round(3).tolist()}}
        nulls = ne.both_nulls(V, k, rng, 2000, taus=TAUS)
        res["nulls"] = {str(t): {key: nulls[t][key] for key in ("L", "D")} for t in TAUS}
        routers = {}
        if name == "two_specialists":
            Xt = stack_embeddings(E, ids, dom, "test")
            idg_v, _ = in_domain("geometry3k", "validation"); idm_v, _ = in_domain("mathvista", "validation")
            have_val = complete(cross, "mathvista", sm[0], "geometry3k", "validation", idg_v) and complete(cross, "geometry3k", sg[0], "mathvista", "validation", idm_v)
            if have_val:
                vids, Cv, domv, kv = union_counts(sg, sm, "validation", cross); Rv = Cv / kv
                asg_knn, asg_lin = fit_routers(stack_embeddings(E, vids, domv, "validation"), Rv, Xt); res["router_fit"] = "validation"; res["n_fit"] = int(len(vids))
            else:
                asg_knn = np.zeros(n, dtype=int); asg_lin = np.zeros(n, dtype=int); fold = np.zeros(n, dtype=int)
                for d in (0, 1):
                    idx = np.flatnonzero(dom == d); fold[idx[rng.permutation(len(idx))]] = np.arange(len(idx)) % N_FOLDS
                for f in range(N_FOLDS):
                    tr, te = fold != f, fold == f
                    asg_knn[te], asg_lin[te] = fit_routers(Xt[tr], R[tr], Xt[te])
                res["router_fit"] = "cross-fit-test"; res["n_folds"] = N_FOLDS
            routers = {"domain": dom.copy(), "embedding nearest-neighbor": asg_knn, "embedding linear": asg_lin}
        res["routing"] = {}
        for t in TAUS:
            need = ne.need(t, k); clear = C >= need
            orac = clear.any(1).mean(); best_m = int(clear.mean(0).argmax()); best = clear.mean(0).max()
            mix = ((C.sum(1) / (M * k)) >= t - 1e-12).mean()
            floor = (C >= ne.need(min(M * t, 1.0 + 1e-9), k)).any(1).mean() if M * t <= 1 else 0.0   # C_orac(M tau)
            ws = [(w, 1 - w) for w in np.linspace(0, 1, 101)] if M == 2 else [np.eye(M)[i] for i in range(M)] + [np.ones(M) / M]
            best_w = max(((np.asarray(w) @ R.T) >= t - 1e-12).mean() for w in ws)
            D = float((clear.any(1) & ~clear.all(1)).mean()); L = float(orac - best)
            cell = {"tau": t, "oracle": float(orac), "best_member": float(best), "best_member_index": best_m, "mixture": float(mix), "floor_orac_Mtau": float(floor),
                    "best_fixed_weighting": float(best_w), "L": L, "D": D, "L_over_D": (L / D if D > 0 else None)}
            for rname, asg in routers.items():
                rr = C[np.arange(n), asg] >= need; cov_r = rr.mean(); gain = cov_r - best
                dis = clear.any(1) & ~clear.all(1); eD = float(np.mean(~rr[dis])) if dis.any() else 0.0
                boots = [rr[ii].mean() - clear[ii, best_m].mean() for ii in (rng.integers(0, n, n) for _ in range(2000))]
                cell[rname] = {"coverage": float(cov_r), "gain": float(gain), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
                               "eD": eD, "fraction_of_L": (float(gain / L) if L > 1e-9 else None)}
            res["routing"][str(t)] = cell
            nl = nulls[t]["L"]
            line = (f"{name:15s} M={M} tau={t:.2f} L={L:.3f} D={D:.3f} redeal={nl['redeal']['null_mean']:.3f} p={nl['redeal']['p']:.3f} margin={nl['margin']['null_mean']:.3f} p_m={nl['margin']['p']:.3f} "
                    f"| best={best:.3f} mix={mix:.3f} floor={floor:.3f} bestw={best_w:.3f} orac={orac:.3f}")
            for rname in routers: line += f" | {rname[:6]} {cell[rname]['gain']:+.3f} [{cell[rname]['lo']:+.3f},{cell[rname]['hi']:+.3f}] eD={cell[rname]['eD']:.2f}"
            print(line)
        if dropped: print(f"{name}: dropped {dropped} (incomplete cross scores)")
        if name == "two_specialists": print(f"{name}: router_fit={res['router_fit']}")
        out["pools"][name] = res
    os.makedirs(os.path.dirname(os.path.abspath(PATHS["out"])), exist_ok=True)
    json.dump(out, open(PATHS["out"], "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
