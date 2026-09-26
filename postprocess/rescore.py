#!/usr/bin/env python3
"""Re-score stored responses with the current checker."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, glob, json, os, sys

import csm_verify as V                                            # noqa: E402

SCRIPT_VERSION = "rescore/1.1"


def main() -> int:
    out = os.environ.get("OUT", "rescore.csv")
    item = os.environ.get("ITEM", "").strip()
    root = os.path.join(os.environ.get("DATA_DIR", ""), "CSM", "rollouts_raw")
    d = os.path.join(root, item)
    if not os.path.isdir(d):
        print(f"FATAL: no such rollout dir {d}")
        return 1

    # Some rollout directories hold MORE THAN ONE generation run over the same
    # prompts. base__geometry3k__val carries eight shards: 0-3 are the 1024-token
    # run (about 1% of completions truncated, max ~2900 chars) and 4-7 are the
    # superseded 512-token run (about 31% truncated, hard-capped near 2050).
    # Reading both silently doubles that arm and mixes a truncated run into the
    # result, so keep only the largest-budget run for each prompt set.
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

    keep, dropped = [], []
    for f in files:
        pids, mx = prof[f]
        # Another file covering the same prompts with materially longer
        # completions means this one is the superseded shorter-budget run.
        superseded = any(g != f and prof[g][0] == pids and prof[g][1] > mx * 1.15
                         for g in files)
        (dropped if superseded else keep).append(f)
    if dropped:
        print(f"note: dropped {len(dropped)} superseded shorter-budget shard(s): "
              + ", ".join(os.path.basename(x) for x in dropped))

    rows, n_roll, n_correct_tot, changed = [], 0, 0, 0
    old_tot = 0
    seen = set()
    for f in keep:
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["model"], r["split"], r["prompt_id"])
            if key in seen:
                print(f"FATAL: duplicate {key} survived shard selection")
                return 1
            seen.add(key)
            comps = r.get("completions")
            if isinstance(comps, str):
                comps = [comps]
            comps = comps or []
            old = r.get("verdicts") or []
            n_ok = 0
            for i, comp in enumerate(comps):
                good = V.verify(V.extract_answer(comp), r.get("gold"),
                                r.get("dataset") or "geometry3k")
                n_ok += bool(good)
                # v1.0 recorded a rule string per draw; anything other than a
                # bare rejection means it had counted that draw as correct.
                if i < len(old):
                    was = old[i] not in ("no_rule_matched", "no_answer_extracted",
                                         "empty_after_normalization", "no_gold")
                    old_tot += was
                    changed += (was != bool(good))
            n_roll += len(comps)
            n_correct_tot += n_ok
            rows.append({"prompt_id": r["prompt_id"], "model": r["model"],
                         "split": r["split"], "dataset": r["dataset"],
                         "n_draws": len(comps), "n_correct": n_ok,
                         "gen_tokens": r.get("gen_tokens", 0)})

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prompt_id", "model", "split", "dataset",
                                           "n_draws", "n_correct", "gen_tokens"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    rate = n_correct_tot / n_roll if n_roll else 0.0
    old_rate = old_tot / n_roll if n_roll else 0.0
    print(f"run={SCRIPT_VERSION} verifier={V.SCRIPT_VERSION} item={item} "
          f"prompts={len(rows)} rollouts={n_roll}")
    print(f"rate_new={rate:.4f} rate_old={old_rate:.4f} "
          f"delta={rate - old_rate:+.4f} verdicts_changed={changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
