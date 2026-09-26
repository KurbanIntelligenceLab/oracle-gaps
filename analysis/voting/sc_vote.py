#!/usr/bin/env python3
"""Majority-vote arms (pooled and single-policy) from stored responses."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, glob, json, os, sys
import numpy as np

import csm_verify as V                                            # noqa: E402

SCRIPT_VERSION = "sc_vote/1.0"
DATASET = os.environ.get("CSM_DATASET", "geometry3k")
SEEDS = ["seed1", "seed2", "seed3", "seed4", "seed5"]
VOTES = [1, 3, 5, 7, 9]
N_MC = int(os.environ.get("SC_N_MC", "4000"))
RNG_SEED = 0


def select_shards(d: str) -> list[str]:
    """Keep only the largest-budget generation run per prompt set (see rescore.py)."""
    files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
    prof = {}
    for f in files:
        pids, mx = set(), 0
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pids.add(r["prompt_id"])
            for x in (r.get("completions") or []):
                mx = max(mx, len(x))
        prof[f] = (pids, mx)
    keep = [f for f in files
            if not any(g != f and prof[g][0] == prof[f][0] and prof[g][1] > prof[f][1] * 1.15
                       for g in files)]
    if len(keep) != len(files):
        print(f"note: {d}: dropped {len(files) - len(keep)} superseded shorter-budget shard(s)")
    return keep


def load_arm(root: str, model: str, split: str) -> dict[str, dict]:
    """prompt_id -> {gold, keys: list[key|None], ok: list[bool]} for one arm."""
    d = os.path.join(root, f"{model}__{DATASET}__{split}")
    if not os.path.isdir(d):
        raise FileNotFoundError(d)
    out = {}
    for f in select_shards(d):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r["prompt_id"]
            if pid in out:
                raise RuntimeError(f"duplicate {model}/{split}/{pid}")
            comps = r.get("completions") or []
            if isinstance(comps, str):
                comps = [comps]
            keys, oks = [], []
            for c in comps:
                ans = V.extract_answer(c) if c else ""
                if not ans.strip():
                    keys.append(None); oks.append(False)
                    continue
                num = V.to_number(ans)
                key = ("n", round(num, 6)) if num is not None else ("s", V.normalize_text(ans))
                keys.append(key)
                oks.append(bool(V.verify(c, r.get("gold"), r.get("dataset") or DATASET)))
            out[pid] = {"gold": r.get("gold"), "keys": keys, "ok": oks, "n": len(comps)}
    return out


def mc_rate(draw_keys: np.ndarray, draw_ok: np.ndarray, v: int, rng) -> int:
    """Number of Monte-Carlo trials (out of N_MC) in which the plurality of v
    uniform draws from `draw_keys` is a correct answer. Key -1 = abstain."""
    n = draw_keys.shape[0]
    K = int(draw_keys.max()) + 1 if n else 0
    idx = rng.integers(0, n, size=(N_MC, v))
    sk = draw_keys[idx]                                  # (N_MC, v) keys, -1 abstain
    counts = np.zeros((N_MC, K + 1), dtype=np.float64)   # column K holds abstentions
    col = np.where(sk < 0, K, sk)
    np.add.at(counts, (np.repeat(np.arange(N_MC), v), col.ravel()), 1.0)
    counts[:, K] = -1.0                                  # abstentions never win
    counts[:, :K] += rng.random((N_MC, K)) * 0.5         # random tie-break, < 1 vote
    win = counts.argmax(axis=1)
    any_vote = (col != K).any(axis=1)
    return int((draw_ok_by_key(draw_keys, draw_ok)[win] & any_vote).sum())


def draw_ok_by_key(draw_keys: np.ndarray, draw_ok: np.ndarray) -> np.ndarray:
    """Correctness of each key id (a key is correct if any draw carrying it verified)."""
    K = int(draw_keys.max()) + 1 if draw_keys.size else 0
    ok = np.zeros(K + 1, dtype=bool)
    for k, o in zip(draw_keys, draw_ok):
        if k >= 0 and o:
            ok[k] = True
    return ok


def main() -> int:
    out = os.environ.get("OUT", "sc_vote.csv")
    split = os.environ.get("ITEM", "").strip()
    if split not in ("val", "test"):
        print(f"FATAL: ITEM must be val or test, got {split!r}")
        return 1
    root = os.path.join(os.environ.get("DATA_DIR", ""), "CSM", "rollouts_raw")

    arms = {m: load_arm(root, m, split) for m in ["base"] + SEEDS}
    pids = sorted(set.intersection(*(set(a) for a in arms.values())))
    n_draw = {arms[m][p]["n"] for m in arms for p in pids}
    if len(n_draw) != 1:
        print(f"FATAL: unequal draw counts across arms/prompts: {sorted(n_draw)}")
        return 1
    print(f"run={SCRIPT_VERSION} verifier={V.SCRIPT_VERSION} split={split} "
          f"prompts={len(pids)} draws_per_arm={n_draw.pop()} n_mc={N_MC} votes={VOTES}")

    rng = np.random.default_rng(RNG_SEED)
    rows, summary = [], {}
    v1_gap = 0.0
    for p in pids:
        # one small integer id per distinct answer key, shared across arms
        keymap = {}
        def enc(k):
            if k is None:
                return -1
            return keymap.setdefault(k, len(keymap))
        per = {m: (np.array([enc(k) for k in arms[m][p]["keys"]]),
                   np.array(arms[m][p]["ok"], dtype=bool)) for m in arms}
        arm_sets = {f"scpool": SEEDS, "scpoolb": ["base"] + SEEDS}
        arm_sets.update({f"sc{m}": [m] for m in SEEDS})
        for arm, members in arm_sets.items():
            dk = np.concatenate([per[m][0] for m in members])
            do = np.concatenate([per[m][1] for m in members])
            for v in VOTES:
                c = mc_rate(dk, do, v, rng)
                rows.append({"prompt_id": p, "model": f"{arm}_v{v}", "split": split,
                             "dataset": DATASET, "n_draws": N_MC, "n_correct": c,
                             "gen_tokens": 0})
                summary.setdefault((arm, v), []).append(c / N_MC)
                if v == 1 and arm.startswith("scseed"):
                    v1_gap = max(v1_gap, abs(c / N_MC - float(per[members[0]][1].mean())))

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prompt_id", "model", "split", "dataset",
                                           "n_draws", "n_correct", "gen_tokens"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"check_v1_max_abs_gap={v1_gap:.4f} (expect ~ sqrt(.25/{N_MC}) = {np.sqrt(.25 / N_MC):.4f})")
    for (arm, v), rates in sorted(summary.items()):
        print(f"arm={arm} v={v} mean_rate={np.mean(rates):.4f} "
              f"cov@.05={np.mean(np.array(rates) >= .05):.4f} "
              f"cov@.30={np.mean(np.array(rates) >= .30):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
