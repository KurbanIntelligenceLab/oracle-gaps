#!/usr/bin/env python3
"""Score the weight-averaged policy beside the members and the budget-matched mixture."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, glob, json, os, sys
import numpy as np

SCRIPT_VERSION = "soup_eval/1.0"
ROOT = str(_REPO_ROOT)
BANDS = {"geometry3k": (0.05, 0.30, (0.05, 0.10, 0.20, 0.30)), "mathvista": (0.50, 0.90, (0.5, 0.6, 0.7, 0.8, 0.9))}
SPLIT = {"val": "validation", "validation": "validation", "test": "test"}


def rauc(p, lo, hi, n_grid=2001):
    grid = np.linspace(lo, hi, n_grid); curve = (np.asarray(p)[None, :] >= grid[:, None]).mean(1)
    tr = getattr(np, "trapezoid", None) or np.trapz
    return float(tr(curve, grid) / (hi - lo))


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "rollout_soup", "sh_*.csv"))):
        rows += list(csv.DictReader(open(f)))
    out = {"script_version": SCRIPT_VERSION, "n_rows": len(rows), "sets": {}}
    rng = np.random.default_rng(0)
    for ds, (lo, hi, taus) in BANDS.items():
        V = np.load(os.path.join(ROOT, "data", ds, "verdicts.npz"))
        Rz = np.load(os.path.join(ROOT, "data", ds, "rates.npz"))
        names = list(Rz["prior_split_names"]); pa, pb = Rz["prior_alpha"], Rz["prior_beta"]
        for split in ("test", "validation"):
            sub = {r["prompt_id"]: (int(r["n_correct"]), int(r["n_draws"])) for r in rows if r["dataset"] == ds and SPLIT[r["split"]] == split}
            m = V["split"] == split; ids = V["prompt_ids"][m]; Vm = V["verdicts"][m]; n, M, k = Vm.shape
            if not sub:
                out["sets"][f"{ds}_{split}"] = {"missing": True}; continue
            missing = [p for p in ids if p not in sub]
            assert not missing, f"{ds} {split}: {len(missing)} prompts without soup rollouts"
            sc = np.array([sub[p][0] for p in ids]); kd = np.array([sub[p][1] for p in ids]); assert (kd == k).all()
            a, b = float(pa[names.index(split)]), float(pb[names.index(split)])
            shrink = lambda s_, kk: (s_ + a) / (kk + a + b)
            Rm = shrink(Vm.sum(2), k); Rs = shrink(sc, k)
            r_members = np.array([rauc(Rm[:, j], lo, hi) for j in range(M)]); r_soup = rauc(Rs, lo, hi)
            m_in = int(r_members.argmax())
            other = "validation" if split == "test" else "test"
            mo = V["split"] == other; Vo = V["verdicts"][mo]; ao, bo = float(pa[names.index(other)]), float(pb[names.index(other)])
            Ro = (Vo.sum(2) + ao) / (Vo.shape[2] + ao + bo); m_sel = int(np.argmax([rauc(Ro[:, j], lo, hi) for j in range(M)]))
            mix_pooled = rauc(Rm.mean(1), lo, hi)                                  # mean of shrunk member rates
            pooled = Vm.reshape(n, M * k)
            subs = np.stack([pooled[np.arange(n)[:, None], np.argsort(rng.random((n, M * k)), 1)[:, :k]].sum(1) for _ in range(200)])
            R16 = shrink(subs, k); mix_eq = float(np.mean([rauc(R16[i], lo, hi) for i in range(200)]))
            d_in, d_sel, d_mix = [], [], []
            for _ in range(2000):
                ii = rng.integers(0, n, n)
                rs = rauc(Rs[ii], lo, hi)
                d_in.append(rs - rauc(Rm[ii, m_in], lo, hi)); d_sel.append(rs - rauc(Rm[ii, m_sel], lo, hi))
                d_mix.append(rs - np.mean([rauc(R16[i, ii], lo, hi) for i in range(0, 200, 20)]))
            ci = lambda d: [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            out["sets"][f"{ds}_{split}"] = dict(
                n=int(n), k=int(k), prior=[a, b], soup_rauc=r_soup, soup_pass1=float(sc.mean() / k), members_rauc=r_members.tolist(),
                members_pass1=[float(x) for x in Vm.mean((0, 2))], best_in_sample=m_in + 1, best_other_split=m_sel + 1,
                mix_pooled_rauc=mix_pooled, mix_equalized_rauc=mix_eq, mix_pass1=float(Vm.mean()),
                soup_minus_best_in=[r_soup - r_members[m_in]] + ci(d_in), soup_minus_best_sel=[r_soup - r_members[m_sel]] + ci(d_sel),
                soup_minus_mix_eq=[r_soup - mix_eq] + ci(d_mix),
                coverage_soup={str(t): float((Rs >= t).mean()) for t in taus},
                coverage_best_member={str(t): float((Rm >= t).mean(0).max()) for t in taus},
                coverage_mix_pooled={str(t): float((Rm.mean(1) >= t).mean()) for t in taus})
            g = out["sets"][f"{ds}_{split}"]
            print(f"{SCRIPT_VERSION} {ds} {split}: soup rAUC={r_soup:.3f} pass1={g['soup_pass1']:.3f} | best member {max(r_members):.3f} (seed {m_in+1}) | "
                  f"mixture pooled {mix_pooled:.3f} equalized {mix_eq:.3f} | soup-best in {g['soup_minus_best_in'][0]:+.3f} [{g['soup_minus_best_in'][1]:+.3f},{g['soup_minus_best_in'][2]:+.3f}] "
                  f"sel {g['soup_minus_best_sel'][0]:+.3f} [{g['soup_minus_best_sel'][1]:+.3f},{g['soup_minus_best_sel'][2]:+.3f}] | soup-mix_eq {g['soup_minus_mix_eq'][0]:+.3f}")
    json.dump(out, open(os.path.join(ROOT, "analysis", "soup_eval.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
