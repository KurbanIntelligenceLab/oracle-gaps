"""Enumerate the canonical prompt identifiers of a dataset."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules

import argparse
import json
import sys
from pathlib import Path

SCRIPT_VERSION = "csm_prompt_ids/1.0"

# The pool a dataset's splits are drawn from. We partition the union of the
# official splits: the paper reports Cover@tau, not leaderboard accuracy, so
# comparability with published splits buys nothing, while one hashed partition
# over one pool guarantees disjoint splits.
POOL_SPLITS = {
    "chartqa": ["train", "val", "test"],
    "geometry3k": ["train", "validation", "test"],
}


def enumerate_ids(inspect_json: str, dataset: str) -> tuple[list[str], dict]:
    rep = json.loads(Path(inspect_json).read_text())
    entry = next((d for d in rep["datasets"] if d.get("dataset") == dataset), None)
    if entry is None:
        raise KeyError(f"dataset {dataset!r} not in {inspect_json}")
    if "error" in entry:
        raise ValueError(f"{dataset}: {entry['error']}")

    ids: list[str] = []
    counts = {}
    for split in POOL_SPLITS[dataset]:
        s = entry["splits"].get(split)
        if s is None or "error" in s:
            raise ValueError(f"{dataset}/{split} unavailable: {s}")
        n = int(s["n_rows_total"])
        counts[split] = n
        ids.extend(f"{dataset}-{split}-{i:06d}" for i in range(n))

    if len(set(ids)) != len(ids):
        raise AssertionError("prompt ids are not unique")
    return ids, counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect", required=True)
    ap.add_argument("--dataset", required=True, choices=sorted(POOL_SPLITS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    ids, counts = enumerate_ids(a.inspect, a.dataset)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + "\n")

    print(f"run={SCRIPT_VERSION} dataset={a.dataset} n_ids={len(ids)} "
          f"counts={counts} out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
