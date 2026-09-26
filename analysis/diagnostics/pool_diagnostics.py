"""In-text diagnostics: seed similarity and chain agreement, coupled draws, the leading validation seed,
and Benjamini-Hochberg families. Usage: python pool_diagnostics.py [similarity|coupling|leader|bh ...]"""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import glob, json, os
import numpy as np
import nulls_extra as ne

ROOT = str(_REPO_ROOT)
DATA = os.path.join(ROOT, "data")


def split_verdicts(ds, split, dtype=np.uint8):
    z = np.load(os.path.join(DATA, ds, "verdicts.npz")); m = z["split"] == split
    return z, m, z["verdicts"][m].astype(dtype)


def similarity():
    _, _, V = split_verdicts("geometry3k", "test"); n, M, k = V.shape; R = V.mean(2)
    off = np.corrcoef(R.T)[np.triu_indices(M, 1)]; rng = np.random.default_rng(0)
    nullc = [np.corrcoef((ne.redeal_counts(V, rng) / k).T)[np.triu_indices(M, 1)].mean() for _ in range(300)]
    print(f"pairwise rate correlation {off.mean():.3f} | re-deal null {np.mean(nullc):.3f} [{np.percentile(nullc, 5):.3f}, {np.percentile(nullc, 95):.3f}]")
    big = (R.max(1) - R.min(1) > 0.25).mean(); nn = []
    for _ in range(200):
        Rn = ne.redeal_counts(V, rng) / k; nn.append((Rn.max(1) - Rn.min(1) > 0.25).mean())
    print(f"prompts with rate spread > 0.25: {100 * big:.1f}% | null {100 * np.mean(nn):.1f}% [{100 * np.percentile(nn, 5):.1f}, {100 * np.percentile(nn, 95):.1f}]")
    C = V.sum(2)
    for seed in (11, 22):
        Ls = [ne.stats(s, k, (0.10,))[0.10]["L"] for s in ne.margin_chain(C, k, np.random.default_rng(seed), 2000)]
        print(f"margin chain seed {seed}: null mean L(0.10) {np.mean(Ls):.4f}, p {np.mean(np.asarray(Ls) >= 0.1125 - 1e-12):.3f}")


def coupling():
    rng = np.random.default_rng(0)
    for ds in ("geometry3k", "mathvista"):
        for split in ("validation", "test"):
            z, m, V = split_verdicts(ds, split, int); T = z["n_gen_tokens"][m]; n, M, k = V.shape
            same, shuf, cl, cs = [], [], [], []
            for a in range(M):
                for b in range(a + 1, M):
                    same.append((V[:, a, :] == V[:, b, :]).mean()); perm = rng.permutation(k); shuf.append((V[:, a, :] == V[:, b, perm]).mean())
            for a in range(M):
                for b in range(a + 1, M):
                    x = T[:, a, :].ravel().astype(float); y = T[:, b, :].ravel().astype(float); cl.append(np.corrcoef(x, y)[0, 1])
                    perm = rng.permutation(k); cs.append(np.corrcoef(x, T[:, b, perm].ravel().astype(float))[0, 1])
            C = V.sum(2); S = C.sum(1); mid = (S > 0) & (S < M * k); obs_var = C[mid].var(1, ddof=0).sum()
            exp_var = np.array([ne.redeal_counts(V, rng)[mid].var(1, ddof=0).sum() for _ in range(300)])
            print(f"{ds} {split}: same-index agreement {np.mean(same):.4f} vs shuffled {np.mean(shuf):.4f} | length corr {np.mean(cl):+.3f} vs {np.mean(cs):+.3f}"
                  f" | all members at the token cap {(T.max(2) >= 1024).all(1).mean():.2f} | count dispersion obs/null {obs_var / exp_var.mean():.3f}")


def leader():
    rng = np.random.default_rng(0); _, _, V = split_verdicts("geometry3k", "validation", int)
    n, M, k = V.shape; C = V.sum(2); need = ne.need(0.10, k); clear = C >= need; cov = clear.mean(0); lead = int(cov.argmax())
    null = np.array([(c >= need)[:, lead].mean() for c in ne.margin_chain(C, k, rng, 400)])
    print(f"leading seed {lead + 1}: coverage {cov[lead]:.3f} | margin-null mean {null.mean():.3f} | P[null <= observed] {(null <= cov[lead]).mean():.3f}")


def bh():
    def adjust(ps):
        ps = np.asarray(ps, float); m = len(ps); order = np.argsort(ps); adj = np.empty(m); prev = 1.0
        for rank, idx in zip(range(m, 0, -1), order[::-1]):
            prev = min(prev, ps[idx] * m / rank); adj[idx] = prev
        return adj
    tests = []
    for f in sorted(glob.glob(os.path.join(ROOT, "analysis", "nulls_extra", "*.json"))):
        name = os.path.basename(f)[:-5]; J = json.load(open(f))
        if name == "two_family_test" or "nulls" not in J: continue
        for t, v in J["nulls"].items():
            tests += [("L", f"{name}@{t}", v["L"]["redeal"]["p"]), ("L", f"{name}@{t}", v["L"]["margin"]["p"])]
            if "D" in v: tests.append(("D", f"{name}@{t}", v["D"]["redeal"]["p"]))
    P = json.load(open(os.path.join(ROOT, "analysis", "specialist_pool.json")))["pools"]
    for pool in ("two_specialists", "all_seeds"):
        for t, v in P[pool]["nulls"].items():
            tests += [("L", f"{pool}@{t}", v["L"]["redeal"]["p"]), ("L", f"{pool}@{t}", v["L"]["margin"]["p"]), ("D", f"{pool}@{t}", v["D"]["redeal"]["p"])]
    adj = adjust([p for _, _, p in tests])
    surv = [(lab, fam, round(float(a), 3)) for (fam, lab, _), a in zip(tests, adj) if a <= 0.05]
    print(f"Benjamini-Hochberg over {len(tests)} tests: {len(surv)} survive at 0.05 -> {surv}")


if __name__ == "__main__":
    parts = _sys.argv[1:] or ["similarity", "coupling", "leader", "bh"]
    for part in parts:
        print(f"== {part}"); globals()[part]()
