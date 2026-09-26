#!/usr/bin/env python3
"""Build the analysis inputs (verdicts, smoothed rates, router assignments) for one dataset."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, csv, json, os, sys
import numpy as np

import csm_rollouts as CR, csm_metrics as M                      # noqa: E402

SCRIPT_VERSION = "assemble_data/1.3"
MEMBERS = ["seed1", "seed2", "seed3", "seed4", "seed5"]
SPLIT_NAME = {"val": "validation", "test": "test"}


def read_export(path, dataset):
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("dataset") and r["dataset"] != dataset:
                continue
            key = (r["split"], r["prompt_id"], r["model"])
            rows.setdefault(key, {})[int(r["draw"])] = (int(r["verdict"]), int(r["chars"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--export", required=True, help="merged export_verdicts csv")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--routers", default=None, help="e7_routers 4.3 json")
    ap.add_argument("--token-caches", nargs="*", default=None,
                    help="caches carrying gen_tokens when --cache lacks them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-map", default="", help="raw split labels to schema names, e.g. testk64=test,b3val=val")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"],
                    help="splits to assemble; test-only extensions pass --splits test")
    a = ap.parse_args()
    split_map = dict(x.split("=") for x in a.split_map.split(",") if x)
    os.makedirs(a.out, exist_ok=True)
    ex = read_export(a.export, a.dataset)
    if split_map:
        ex = {(split_map.get(sp, sp), pid, m): v for (sp, pid, m), v in ex.items()}

    V_all, B_all, T_all, C_all, pid_all, split_all = [], [], [], [], [], []
    priors = {}
    R_all, b_all = [], []
    for split in a.splits:
        pids, succ, draws = CR.rate_matrix(a.cache, split=split, models=["base"] + MEMBERS)
        k = int(draws.max()); assert draws.min() == k, "unequal draw counts"
        n = len(pids)
        V = np.zeros((n, len(MEMBERS), k), np.uint8); Bv = np.zeros((n, k), np.uint8)
        Cc = np.zeros((n, len(MEMBERS), k), np.int32)
        for i, p in enumerate(pids):
            for j, m in enumerate(["base"] + MEMBERS):
                d = ex.get((split, p, m))
                if d is None or len(d) != k:
                    print(f"FATAL: export lacks {split}/{p}/{m} ({0 if d is None else len(d)} draws)"); return 1
                v = np.array([d[t][0] for t in range(k)], np.uint8)
                if int(v.sum()) != int(succ[j, i]):
                    print(f"FATAL: {split}/{p}/{m}: export {int(v.sum())} correct vs cache {int(succ[j, i])}"); return 1
                if j == 0:
                    Bv[i] = v
                else:
                    V[i, j - 1] = v; Cc[i, j - 1] = [d[t][1] for t in range(k)]
        # tokens per prompt (summed over draws) from the cache(s) that carry them
        tok = {}
        for c in [a.cache] + (a.token_caches or []):
            for r in CR.load_cache(c, split=split):
                if r.get("gen_tokens") and r["model"] in MEMBERS:
                    tok.setdefault((r["model"], r["prompt_id"]), r["gen_tokens"] / r["n_draws"])
        T = np.zeros((n, len(MEMBERS), k), np.int32); miss = 0
        for i, p in enumerate(pids):
            for j, m in enumerate(MEMBERS):
                t = tok.get((m, p))
                if t is None: miss += 1
                else: T[i, j, :] = int(round(t))
        if miss: print(f"note: {split}: {miss} (prompt, member) cells without token counts")
        # the pipeline's estimator, verbatim: prior pooled over base+members on this split
        al, be = M.eb_beta_binomial_prior(succ, draws)
        Rf = M.posterior_rates(succ, draws)
        priors[SPLIT_NAME[split]] = (al, be)
        V_all.append(V); B_all.append(Bv); T_all.append(T); C_all.append(Cc)
        pid_all += pids; split_all += [SPLIT_NAME[split]] * n
        R_all.append(Rf[1:].T); b_all.append(Rf[0])
        print(f"{a.dataset} {split}: n={n} k={k} prior=({al:.4f},{be:.4f}) strength={al+be:.3f}")

    V = np.concatenate(V_all); Bv = np.concatenate(B_all); T = np.concatenate(T_all); Cc = np.concatenate(C_all)
    pid = np.array(pid_all, dtype="<U64"); spl = np.array(split_all, dtype="<U16")
    mem = np.array(MEMBERS, dtype="<U64")
    np.savez(os.path.join(a.out, "verdicts.npz"), verdicts=V, base_verdicts=Bv, prompt_ids=pid,
             member_ids=mem, split=spl, n_gen_tokens=T, n_gen_chars=Cc,
             n_gen_tokens_note=np.array("per-prompt mean tokens per draw, broadcast over draws; n_gen_chars is exact per draw"))
    R = np.concatenate(R_all); b = np.concatenate(b_all)
    at, bt = priors["test"]
    names = list(priors)
    np.savez(os.path.join(a.out, "rates.npz"), R=R, b=b, alpha=np.float64(at), beta=np.float64(bt),
             fit_split=np.array("test"), prompt_ids=pid, member_ids=mem, split=spl,
             prior_split_names=np.array(names, dtype="<U16"),
             prior_alpha=np.array([priors[s][0] for s in names]), prior_beta=np.array([priors[s][1] for s in names]),
             estimator=np.array("moment-matched beta-binomial, one prior per evaluated split, pooled over base and members; posterior mean"))
    if a.routers:
        rep = json.load(open(a.routers))
        fams = [kk for kk in ("tfidf_logreg", "gbm_percorrect", "knn_valrate") if kk in rep["routers"] and "per_init" in rep["routers"][kk]]
        n_init = len(rep["inits"])
        C = np.zeros((len(fams), n_init, len(pid)), np.int64)
        is_val = spl == "validation"; is_test = spl == "test"
        # the router job's prompt order is the cache's sorted-common order, same as here
        for f, kk in enumerate(fams):
            for i, d in enumerate(rep["routers"][kk]["per_init"]):
                if is_test.any():
                    C[f, i, is_test] = np.array(d["assignment"])
                if is_val.any():
                    C[f, i, is_val] = np.array(d["assignment_val_in_sample"])
        np.savez(os.path.join(a.out, "routers.npz"), choice=C, router_ids=np.array(fams, dtype="<U64"),
                 val_in_sample=np.array(True), inits=np.array(rep["inits"]))
        print(f"routers: {fams} x {n_init} inits")
    print(f"run={SCRIPT_VERSION} wrote {a.out}: verdicts {V.shape} rates {R.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
