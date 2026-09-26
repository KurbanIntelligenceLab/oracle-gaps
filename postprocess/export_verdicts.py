#!/usr/bin/env python3
"""Export per-response verdicts for the offline analyses."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import csv, glob, json, os, sys

import csm_verify as V                                            # noqa: E402

SCRIPT_VERSION = "export_verdicts/1.0"
MODELS = ["base", "seed1", "seed2", "seed3", "seed4", "seed5"]


def select_shards(d):
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
    return [f for f in files
            if not any(g != f and prof[g][0] == prof[f][0] and prof[g][1] > prof[f][1] * 1.15
                       for g in files)]


def main() -> int:
    out = os.environ.get("OUT", "export_verdicts.csv")
    ds, split = os.environ.get("ITEM", "|").split("|")
    root = os.path.join(os.environ.get("DATA_DIR", ""), "CSM", "rollouts_raw")
    rows, n_prompt = [], {}
    for m in MODELS:
        d = os.path.join(root, f"{m}__{ds}__{split}")
        if not os.path.isdir(d):
            print(f"FATAL: missing {d}"); return 1
        seen = set()
        for f in select_shards(d):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                pid = r["prompt_id"]
                if pid in seen:
                    print(f"FATAL: duplicate {m}/{split}/{pid}"); return 1
                seen.add(pid)
                comps = r.get("completions") or []
                if isinstance(comps, str):
                    comps = [comps]
                for i, c in enumerate(comps):
                    ok = bool(V.verify(c, r.get("gold"), r.get("dataset") or ds)) if c else False
                    rows.append({"prompt_id": pid, "model": m, "split": split, "draw": i,
                                 "verdict": int(ok), "chars": len(c or "")})
        n_prompt[m] = len(seen)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prompt_id", "model", "split", "draw", "verdict", "chars"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"run={SCRIPT_VERSION} verifier={V.SCRIPT_VERSION} dataset={ds} split={split} "
          f"rows={len(rows)} prompts_per_model={n_prompt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
