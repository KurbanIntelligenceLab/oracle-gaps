"""Choose the reliability band from base-model success rates on validation."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

SCRIPT_VERSION = "csm_tau_calibrate/1.0"

CANDIDATE_BANDS = [
    (0.05, 0.30), (0.10, 0.40), (0.15, 0.50), (0.20, 0.60),
    (0.30, 0.70), (0.40, 0.80), (0.50, 0.90),
]


def load(shard_dir: str) -> dict[str, list[float]]:
    by_ds: dict[str, list[float]] = defaultdict(list)
    files = sorted(glob.glob(os.path.join(shard_dir, "sh_*.csv")))
    if not files:
        raise FileNotFoundError(f"no sh_*.csv under {shard_dir}")
    seen: set[tuple[str, str]] = set()
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                key = (r["dataset"], r["prompt_id"])
                if key in seen:           # a re-run shard must not double-count
                    continue
                seen.add(key)
                d, c = int(r["n_draws"]), int(r["n_correct"])
                by_ds[r["dataset"]].append(c / d)
    return dict(by_ds)


def summarize(rates: list[float]) -> dict:
    n = len(rates)
    zero = sum(1 for r in rates if r == 0.0)
    one = sum(1 for r in rates if r == 1.0)
    nondeg = n - zero - one
    out = {
        "n_items": n,
        "mean_rate": sum(rates) / n if n else 0.0,
        "frac_zero": zero / n if n else 0.0,
        "frac_one": one / n if n else 0.0,
        "frac_non_degenerate": nondeg / n if n else 0.0,
        "bands": {},
    }
    for lo, hi in CANDIDATE_BANDS:
        k = sum(1 for r in rates if lo <= r <= hi)
        out["bands"][f"[{lo:.2f},{hi:.2f}]"] = {
            "frac_all_items": k / n if n else 0.0,
            # the share of the USABLE (non-degenerate) items the band captures:
            # the honest denominator, since degenerate items are unreachable
            "frac_of_non_degenerate": k / nondeg if nondeg else 0.0,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default="results/p3_calibrate")
    ap.add_argument("--out", default="analysis/tau_calibration.json")
    a = ap.parse_args(argv)

    by_ds = load(a.shards)
    report = {"script_version": SCRIPT_VERSION, "shard_dir": a.shards, "datasets": {}}

    print(f"run={SCRIPT_VERSION} shards={a.shards}")
    for ds, rates in sorted(by_ds.items()):
        s = summarize(rates)
        report["datasets"][ds] = s
        print(f"\n=== {ds}  n={s['n_items']} ===")
        print(f"  mean_rate={s['mean_rate']:.4f}  never_solved={s['frac_zero']:.3f}  "
              f"always_solved={s['frac_one']:.3f}  NON-DEGENERATE={s['frac_non_degenerate']:.3f}")
        print("  band            frac_all  frac_of_usable")
        for b, v in s["bands"].items():
            print(f"  {b:14s}  {v['frac_all_items']:8.3f}  {v['frac_of_non_degenerate']:14.3f}")

    # recommendation: maximise captured usable mass, on the dataset with the most
    # usable mass to begin with
    best_ds = max(report["datasets"], key=lambda d: report["datasets"][d]["frac_non_degenerate"])
    bands = report["datasets"][best_ds]["bands"]
    best_band = max(bands, key=lambda b: bands[b]["frac_all_items"])
    report["recommendation"] = {
        "primary_dataset": best_ds,
        "band": best_band,
        "rationale": "dataset with the largest non-degenerate fraction; band capturing "
                     "the most items on it",
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)

    print(f"\nRECOMMEND primary_dataset={best_ds} band={best_band}")
    print(f"wrote={a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
