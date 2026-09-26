#!/usr/bin/env python3
"""Cross-dataset pools under stronger training, from their sampled shards."""
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nulls_extra as ne   # noqa: E402
import specialist_pool as sp   # noqa: E402

SCRIPT_VERSION = "specialist_pool_long/1.0"
ROOT = sp.ROOT
CROSS = os.environ.get("CSM_LONG_CROSS", os.path.join(ROOT, "results", "rollout_long"))
OUT = os.environ.get("CSM_LONG_OUT", os.path.join(ROOT, "analysis", "specialist_pool_long.json"))
TAUS = sp.TAUS


def counts():
    """(adapter_ds, seed_token, eval_ds, split) -> {prompt_id: (n_correct, n_draws)}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(CROSS, "sh_*.csv"))):
        for r in csv.DictReader(open(f)):
            ads, seed = r["model"].split("_", 1)
            out.setdefault((ads, seed, r["dataset"], sp.SPLIT[r["split"]]), {})[r["prompt_id"]] = (int(r["n_correct"]), int(r["n_draws"]))
    return out


def union(members, split, C):
    """members: list of (adapter_ds, seed_token). Rows: geometry3k probe then mathvista probe. Returns ids, counts, domain, k."""
    idg, _ = sp.in_domain("geometry3k", split); idm, _ = sp.in_domain("mathvista", split)
    rows, doms, k = [], [], None
    for p in idg:
        rows.append([C[(a, s, "geometry3k", split)][p][0] for a, s in members]); k = k or C[(members[0][0], members[0][1], "geometry3k", split)][p][1]; doms.append(0)
    for p in idm:
        rows.append([C[(a, s, "mathvista", split)][p][0] for a, s in members]); doms.append(1)
    return list(idg) + list(idm), np.array(rows), np.array(doms), k


def main():
    rng = np.random.default_rng(0); C = counts(); E = sp.embeddings()
    adapters = sorted({(a, s) for (a, s, _, _) in C})
    print(f"run={SCRIPT_VERSION} adapters={adapters} keys={len(C)}")
    idg_t, _ = sp.in_domain("geometry3k", "test"); idm_t, _ = sp.in_domain("mathvista", "test"); idg_v, _ = sp.in_domain("geometry3k", "validation"); idm_v, _ = sp.in_domain("mathvista", "validation")
    def complete(a, s):
        return all(p in C.get((a, s, "geometry3k", "test"), {}) for p in idg_t) and all(p in C.get((a, s, "mathvista", "test"), {}) for p in idm_t)
    members_all = [m for m in adapters if complete(*m)]; dropped = [m for m in adapters if m not in members_all]
    geo = [m for m in members_all if m[0] == "geometry3k"]; mv = [m for m in members_all if m[0] == "mathvista"]
    out = {"script_version": SCRIPT_VERSION, "adapters": adapters, "dropped": dropped, "pools": {}}
    pools = {"two_long": (geo[:1] + mv[:1]) if geo and mv else [], "all_long": members_all}
    for name, members in pools.items():
        if len(members) < 2: out["pools"][name] = {"skipped": "fewer than two complete members"}; continue
        ids, Ct, dom, k = union(members, "test", C); n, M = Ct.shape; V = sp.counts_to_V(Ct, k, rng); R = Ct / k
        res = {"n": int(n), "M": int(M), "k": int(k), "members": [f"{a} {s}" for a, s in members],
               "domain_rates": {"geometry3k_prompts": R[dom == 0].mean(0).round(3).tolist(), "mathvista_prompts": R[dom == 1].mean(0).round(3).tolist()}}
        nulls = ne.both_nulls(V, k, rng, 2000, taus=TAUS); res["nulls"] = {str(t): {key: nulls[t][key] for key in ("L", "D")} for t in TAUS}
        routers = {"domain": np.where(dom == 0, [i for i, m in enumerate(members) if m[0] == "geometry3k"][0], [i for i, m in enumerate(members) if m[0] == "mathvista"][0])}
        have_val = all(all(p in C.get((a, s, "geometry3k", "validation"), {}) for p in idg_v) and all(p in C.get((a, s, "mathvista", "validation"), {}) for p in idm_v) for a, s in members)
        if have_val:
            vids, Cv, domv, kv = union(members, "validation", C); Xv = sp.stack_embeddings(E, vids, domv, "validation"); Xt = sp.stack_embeddings(E, ids, dom, "test")
            knn, lin = sp.fit_routers(Xv, Cv / kv, Xt); routers.update({"embedding nearest-neighbor": knn, "embedding linear": lin}); res["router_fit"] = "validation"
        else:
            res["router_fit"] = "validation cross scores incomplete; embedding routers skipped"
        res["routing"] = {}
        for t in TAUS:
            need = ne.need(t, k); clear = Ct >= need; orac = clear.any(1).mean(); best_m = int(clear.mean(0).argmax()); best = clear.mean(0).max()
            mix = ((Ct.sum(1) / (M * k)) >= t - 1e-12).mean(); floor = (Ct >= ne.need(min(M * t, 1.0 + 1e-9), k)).any(1).mean() if M * t <= 1 else 0.0
            ws = [(w, 1 - w) for w in np.linspace(0, 1, 101)] if M == 2 else [np.eye(M)[i] for i in range(M)] + [np.ones(M) / M]
            best_w = max(((np.asarray(w) @ R.T) >= t - 1e-12).mean() for w in ws)
            D = float((clear.any(1) & ~clear.all(1)).mean()); L = float(orac - best)
            cell = {"tau": t, "oracle": float(orac), "best_member": float(best), "best_member_index": best_m, "mixture": float(mix), "floor_orac_Mtau": float(floor), "best_fixed_weighting": float(best_w), "L": L, "D": D, "L_over_D": (L / D if D > 0 else None)}
            for rname, asg in routers.items():
                rr = Ct[np.arange(n), asg] >= need; gain = rr.mean() - best; dis = clear.any(1) & ~clear.all(1); eD = float(np.mean(~rr[dis])) if dis.any() else 0.0
                boots = [rr[ii].mean() - clear[ii, best_m].mean() for ii in (rng.integers(0, n, n) for _ in range(2000))]
                cell[rname] = {"coverage": float(rr.mean()), "gain": float(gain), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "eD": eD, "fraction_of_L": (float(gain / L) if L > 1e-9 else None)}
            res["routing"][str(t)] = cell; nl = nulls[t]["L"]
            line = f"{name:9s} M={M} tau={t:.2f} L={L:.3f} D={D:.3f} redeal={nl['redeal']['null_mean']:.3f} p={nl['redeal']['p']:.3f} margin={nl['margin']['null_mean']:.3f} p_m={nl['margin']['p']:.3f} | best={best:.3f} mix={mix:.3f} floor={floor:.3f} orac={orac:.3f}"
            for rname in routers: line += f" | {rname[:6]} {cell[rname]['gain']:+.3f} [{cell[rname]['lo']:+.3f},{cell[rname]['hi']:+.3f}] eD={cell[rname]['eD']:.2f}"
            print(line)
        print(f"{name}: domain rates geo {res['domain_rates']['geometry3k_prompts']} mv {res['domain_rates']['mathvista_prompts']} | router_fit={res['router_fit']}")
        out["pools"][name] = res
    json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT, "| dropped:", dropped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
