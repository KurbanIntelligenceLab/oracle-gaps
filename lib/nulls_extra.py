#!/usr/bin/env python3
"""Re-deal and margin-preserving nulls, planted specialization, split-half estimates,
the bootstrap excess, and the budget-matched mixture. Run directly on one verdict set."""
import argparse, json, os, sys
import numpy as np

SCRIPT_VERSION = "nulls_extra/1.1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAUS = (0.05, 0.10, 0.20, 0.30)


def need(t, k):
    return int(np.ceil(t * k - 1e-9))


def stats(counts, k, taus=TAUS):
    """counts (n, M) -> tau -> dict(D, L, gap, best, orac, mix). Count thresholds, as Table 1."""
    n, M = counts.shape
    out = {}
    for t in taus:
        clear = counts >= need(t, k)
        orac = clear.any(1).mean(); best = clear.mean(0).max()
        mix = ((counts.sum(1) / (M * k)) >= t - 1e-12).mean()
        D = (clear.any(1) & ~clear.all(1)).mean()
        out[t] = dict(D=float(D), L=float(orac - best), gap=float(orac - mix), best=float(best), orac=float(orac), mix=float(mix))
    return out


def redeal_counts(V, rng):
    """One re-deal: permute each prompt's M*k draws, reshape, count."""
    n, M, k = V.shape
    flat = V.reshape(n, M * k)
    idx = np.argsort(rng.random((n, M * k)), axis=1)
    return flat[np.arange(n)[:, None], idx].reshape(n, M, k).sum(2)


def margin_chain(counts, k, rng, n_samples, burn_factor=60, thin_factor=2):
    """The re-deal law conditioned on the member totals. A re-deal permutes each prompt's M*k
    draws, so a count table has probability proportional to prod_{p,m} C(k, c_pm); keeping the
    column sums (each member's total successes, its marginal quality) fixed and sampling that law
    isolates the prompt-by-member interaction. The chain uses symmetric 2x2 transfers (one success
    from member a to b in prompt p, one from b to a in prompt q) with the Metropolis ratio of
    those weights, so it converges to the conditioned re-deal law and not to the far more
    dispersed uniform law on the fibre."""
    from math import lgamma, exp
    C = counts.copy(); n, M = C.shape
    lb = np.array([lgamma(k + 1) - lgamma(c + 1) - lgamma(k - c + 1) for c in range(k + 1)])
    burn = burn_factor * n * M; thin = thin_factor * n * M
    total = burn + thin * n_samples
    out = []
    P = rng.integers(0, n, size=(total, 2)); A = rng.integers(0, M, size=(total, 2)); U = rng.random(total)
    for i in range(total):
        p, q = P[i]; a, b = A[i]
        if p != q and a != b and C[p, a] > 0 and C[q, b] > 0 and C[p, b] < k and C[q, a] < k:
            cpa, cpb, cqb, cqa = C[p, a], C[p, b], C[q, b], C[q, a]
            d = (lb[cpa - 1] + lb[cpb + 1] + lb[cqb - 1] + lb[cqa + 1]) - (lb[cpa] + lb[cpb] + lb[cqb] + lb[cqa])
            if d >= 0 or U[i] < exp(d):
                C[p, a] -= 1; C[p, b] += 1; C[q, b] -= 1; C[q, a] += 1
        if i >= burn and (i - burn) % thin == 0:
            out.append(C.copy())
    return np.array(out[:n_samples])


def null_summary(obs, null_vals):
    null_vals = np.asarray(null_vals, float)
    return dict(obs=float(obs), null_mean=float(null_vals.mean()), null_lo=float(np.percentile(null_vals, 5)),
                null_hi=float(np.percentile(null_vals, 95)), p=float(np.mean(null_vals >= obs - 1e-12)),
                excess=float(obs - null_vals.mean()))


def both_nulls(V, k, rng, n_perm, taus=TAUS, keys=("L", "D", "gap")):
    counts = V.sum(2); obs = stats(counts, k, taus)
    rd = [stats(redeal_counts(V, rng), k, taus) for _ in range(n_perm)]
    mg = [stats(c, k, taus) for c in margin_chain(counts, k, rng, n_perm)]
    res = {}
    for t in taus:
        res[t] = {}
        for key in keys:
            res[t][key] = dict(redeal=null_summary(obs[t][key], [r[t][key] for r in rd]),
                               margin=null_summary(obs[t][key], [m[t][key] for m in mg]))
    return res


def excess_bootstrap(V, k, rng, n_boot=300, n_perm=200, tau=0.10):
    """Prompt bootstrap of (observed L - re-deal null mean) at one threshold."""
    n = V.shape[0]; vals = []
    for _ in range(n_boot):
        Vb = V[rng.integers(0, n, n)]
        obs = stats(Vb.sum(2), k, (tau,))[tau]["L"]
        nul = np.mean([stats(redeal_counts(Vb, rng), k, (tau,))[tau]["L"] for _ in range(n_perm)])
        vals.append(obs - nul)
    vals = np.asarray(vals)
    return dict(lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)), sd=float(vals.std()))


def plant(V, k, delta, rng):
    """Transfer round(delta*k) successes per prompt from other members to a random specialist,
    keeping the prompt total: interaction without a change in difficulty or in member marginals
    in expectation (each member is the specialist on n/M prompts)."""
    counts = V.sum(2).copy(); n, M = counts.shape
    t = int(round(delta * k))
    spec = rng.integers(0, M, n)
    for p in range(n):
        m = spec[p]; moved = 0
        others = [j for j in range(M) if j != m]
        while moved < t and counts[p, m] < k and any(counts[p, j] > 0 for j in others):
            j = rng.choice([j for j in others if counts[p, j] > 0])
            counts[p, j] -= 1; counts[p, m] += 1; moved += 1
    return counts, spec


def counts_to_V(counts, k, rng):
    n, M = counts.shape
    V = np.zeros((n, M, k), dtype=np.uint8)
    for p in range(n):
        for m in range(M):
            V[p, m, :counts[p, m]] = 1
            V[p, m] = V[p, m][rng.permutation(k)]
    return V


def power_curve(V, k, rng, deltas=(0.0, 0.05, 0.10, 0.15, 0.20, 0.25), reps=20, n_perm=400, tau=0.10):
    rows = []
    for d in deltas:
        rej_rd = rej_mg = 0; eff = []
        for _ in range(reps):
            C, _ = plant(V, k, d, rng); Vp = counts_to_V(C, k, rng)
            obs = stats(C, k, (tau,))[tau]["L"]
            rd = [stats(redeal_counts(Vp, rng), k, (tau,))[tau]["L"] for _ in range(n_perm)]
            mg = [stats(c, k, (tau,))[tau]["L"] for c in margin_chain(C, k, rng, n_perm, burn_factor=30, thin_factor=1)]
            rej_rd += np.mean(np.asarray(rd) >= obs) < 0.05; rej_mg += np.mean(np.asarray(mg) >= obs) < 0.05
            eff.append(obs - np.mean(rd))
        rows.append(dict(delta=d, power_redeal=rej_rd / reps, power_margin=rej_mg / reps, excess_mean=float(np.mean(eff))))
    return rows


def split_half(V, k, rng, reps=200, taus=TAUS):
    """Oracle and comparator both selected on half A and scored on half B (matched), beside the
    paper's version (comparator selected on the scoring half) and the plug-in on half B."""
    n, M, _ = V.shape; h = k // 2; res = {}
    for t in taus:
        matched, paper, plug = [], [], []
        for _ in range(reps):
            idx = rng.permutation(k); a, b = idx[:h], idx[h:]
            Ca, Cb = V[:, :, a].sum(2), V[:, :, b].sum(2)
            nd = need(t, h)
            pick = Ca.argmax(1); hon = (Cb[np.arange(n), pick] >= nd).mean()
            best_a = (Ca >= nd).mean(0).argmax()                   # comparator chosen on half A
            matched.append(hon - (Cb[:, best_a] >= nd).mean())
            paper.append(hon - (Cb >= nd).mean(0).max())           # comparator chosen on half B
            plug.append((Cb >= nd).any(1).mean() - (Cb >= nd).mean(0).max())
        res[t] = dict(half_k=h, plug_in=float(np.mean(plug)), paper=float(np.mean(paper)), matched=float(np.mean(matched)))
    return res


def rauc(p, lo=0.05, hi=0.30, n_grid=2001):
    grid = np.linspace(lo, hi, n_grid); curve = (np.asarray(p)[None, :] >= grid[:, None]).mean(1)
    tr = getattr(np, "trapezoid", None) or np.trapz
    return float(tr(curve, grid) / (hi - lo))


def equalized_mixture(V, k, rng, alpha, beta, band, reps=200, n_boot=1000, V_sel=None):
    """Seeds-only mixture with 16-draw estimator variance: subsample k of the pooled M*k draws.
    Rates are shrunk with the split's prior exactly as rates.npz does."""
    n, M, _ = V.shape; lo, hi = band
    shrink = lambda s, kk: (s + alpha) / (kk + alpha + beta)
    Rm = shrink(V.sum(2), k)                                        # (n, M) member rates
    r_members = np.array([rauc(Rm[:, m], lo, hi) for m in range(M)])
    m_in = int(r_members.argmax())
    m_sel = m_in
    if V_sel is not None:                                            # comparator chosen on another split
        Rs = shrink(V_sel.sum(2), V_sel.shape[2]); m_sel = int(np.argmax([rauc(Rs[:, m], lo, hi) for m in range(M)]))
    pooled = V.reshape(n, M * k)
    mix80 = rauc(shrink(pooled.sum(1), M * k), lo, hi)
    subs = np.stack([pooled[np.arange(n)[:, None], np.argsort(rng.random((n, M * k)), 1)[:, :k]].sum(1) for _ in range(reps)])  # (reps, n)
    R16 = shrink(subs, k)                                            # (reps, n)
    mix16 = float(np.mean([rauc(R16[r], lo, hi) for r in range(reps)]))
    # paired bootstrap over prompts of mix16 - best (fixed subsample set, averaged)
    diffs_in, diffs_sel = [], []
    for _ in range(n_boot):
        ii = rng.integers(0, n, n)
        m16 = np.mean([rauc(R16[r, ii], lo, hi) for r in range(0, reps, 10)])
        diffs_in.append(m16 - rauc(Rm[ii, m_in], lo, hi)); diffs_sel.append(m16 - rauc(Rm[ii, m_sel], lo, hi))
    ci = lambda d: [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return dict(rauc_members=r_members.tolist(), best_in_sample=m_in + 1, best_selected=m_sel + 1,
                mix_pooled=mix80, mix_equalized=mix16, gain_vs_in_sample=mix16 - r_members[m_in], ci_in_sample=ci(diffs_in),
                gain_vs_selected=mix16 - r_members[m_sel], ci_selected=ci(diffs_sel))


def k_trend(V64, rng, ks=(8, 16, 32, 64), reps=40, n_perm=200, tau=0.10, n_boot=100):
    """Excess at tau against the budget k, subsampled from 64 draws, and a fit a + b/sqrt(k)."""
    n, M, K = V64.shape; rows = []
    for kk in ks:
        vals = []
        for _ in range(reps if kk < K else 1):
            sub = np.stack([V64[:, m][:, rng.permutation(K)[:kk]] for m in range(M)], 1)
            obs = stats(sub.sum(2), kk, (tau,))[tau]["L"]
            nul = np.mean([stats(redeal_counts(sub, rng), kk, (tau,))[tau]["L"] for _ in range(n_perm)])
            vals.append((obs, nul))
        vals = np.asarray(vals); rows.append(dict(k=kk, L=float(vals[:, 0].mean()), null=float(vals[:, 1].mean()), excess=float((vals[:, 0] - vals[:, 1]).mean())))
    X = np.array([[1.0, 1 / np.sqrt(r["k"])] for r in rows]); y = np.array([r["excess"] for r in rows])
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    boots = []
    for _ in range(n_boot):
        ii = rng.integers(0, n, n); Vb = V64[ii]; yy = []
        for kk in ks:
            sub = np.stack([Vb[:, m][:, rng.permutation(K)[:kk]] for m in range(M)], 1)
            obs = stats(sub.sum(2), kk, (tau,))[tau]["L"]
            nul = np.mean([stats(redeal_counts(sub, rng), kk, (tau,))[tau]["L"] for _ in range(60)])
            yy.append(obs - nul)
        boots.append(np.linalg.lstsq(X, np.array(yy), rcond=None)[0][0])
    return dict(rows=rows, limit=float(a), slope=float(b), limit_ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])


def load_set(path, split):
    z = np.load(path)
    V = z["verdicts"]; ids = z["prompt_ids"]
    if "split" in z.files and split:
        m = z["split"] == split; V, ids = V[m], ids[m]
    return V.astype(np.uint8), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--verdicts", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--rates", default=None, help="rates.npz of the same data dir (prior for the equalized mixture)")
    ap.add_argument("--select-split", default=None, help="split whose best member is the validation-selected comparator")
    ap.add_argument("--band", type=float, nargs=2, default=(0.05, 0.30))
    ap.add_argument("--family", default=None, help="second verdicts.npz to stack as extra members (two-family pool)")
    ap.add_argument("--n-perm", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", default="", help="comma list of analyses to skip: power,split,mix,ktrend,boot")
    a = ap.parse_args(); skip = set(a.skip.split(",")) - {""}
    rng = np.random.default_rng(a.seed)
    V, ids = load_set(a.verdicts, a.split)
    fam = None
    if a.family:
        V2, ids2 = load_set(a.family, a.split); pos = {p: i for i, p in enumerate(ids2)}
        V2 = V2[[pos[p] for p in ids]]; fam = dict(m1=V.shape[1], m2=V2.shape[1], rate1=float(V.mean()), rate2=float(V2.mean()))
        V = np.concatenate([V, V2], 1)
    n, M, k = V.shape
    out = dict(script_version=SCRIPT_VERSION, name=a.name, n=n, M=M, k=k, family=fam)
    out["nulls"] = {str(t): v for t, v in both_nulls(V, k, rng, a.n_perm).items()}
    if "boot" not in skip:
        out["excess_ci_0.10"] = excess_bootstrap(V, k, rng)
    if "power" not in skip:
        out["power"] = power_curve(V, k, rng)
    if "split" not in skip:
        out["split_half"] = {str(t): v for t, v in split_half(V, k, rng).items()}
    if "mix" not in skip and a.rates:
        z = np.load(a.rates); names = list(z["prior_split_names"]) if "prior_split_names" in z.files else []
        if names and a.split in names:
            i = names.index(a.split); alpha, beta = float(z["prior_alpha"][i]), float(z["prior_beta"][i])
        else:
            alpha, beta = float(z["alpha"]), float(z["beta"])
        Vsel = load_set(a.verdicts, a.select_split)[0] if a.select_split else None
        out["mixture"] = dict(alpha=alpha, beta=beta, **equalized_mixture(V, k, rng, alpha, beta, tuple(a.band), V_sel=Vsel))
    if "ktrend" not in skip and k >= 64:
        out["k_trend"] = k_trend(V, rng)
    os.makedirs(os.path.join(ROOT, "analysis", "nulls_extra"), exist_ok=True)
    path = os.path.join(ROOT, "analysis", "nulls_extra", a.name + ".json")
    json.dump(out, open(path, "w"), indent=1)
    t = "0.1"; L = out["nulls"][t]["L"]; D = out["nulls"][t]["D"]
    print(f"{SCRIPT_VERSION} name={a.name} n={n} M={M} k={k} L(0.10)={L['redeal']['obs']:.3f} "
          f"redeal_null={L['redeal']['null_mean']:.3f} p={L['redeal']['p']:.3f} excess={L['redeal']['excess']:+.3f} | "
          f"margin_null={L['margin']['null_mean']:.3f} p={L['margin']['p']:.3f} excess={L['margin']['excess']:+.3f} | "
          f"D(0.10)={D['redeal']['obs']:.3f} p_redeal={D['redeal']['p']:.3f} p_margin={D['margin']['p']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
