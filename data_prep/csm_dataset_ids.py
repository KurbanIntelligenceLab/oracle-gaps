"""Enumerate prompt identifiers and split groups for the second dataset."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import glob, json, os, sys

SCRIPT_VERSION = "csm_dataset_ids/1.0"
SPEC = {
    "mathvista": {"dir": "data", "split": "testmini", "pattern": "testmini-*.parquet",
                  "cols": ["metadata", "question_type", "answer_type"],
                  "exclude_source": {"geometry3k", "geometry3K", "Geometry3K", "GEOS"}},
    "mathverse": {"dir": "", "split": "testmini", "pattern": "testmini.parquet",
                  "cols": ["problem_index", "problem_version", "question_type", "metadata"],
                  "exclude_source": {"Geometry3K"},
                  "group_col": "problem_index"},
}


def main() -> int:
    import pyarrow.parquet as pq
    ds = os.environ.get("ITEM", "").strip()
    out = os.environ.get("OUT", f"{ds}_ids.json")
    if ds not in SPEC:
        print(f"FATAL: unknown dataset {ds!r}"); return 1
    sp = SPEC[ds]
    root = os.path.join(os.environ.get("DATA_DIR", ""), "CSM", "datasets", ds)
    base = os.path.join(root, sp["dir"]) if sp["dir"] else root
    files = sorted(glob.glob(os.path.join(base, sp["pattern"])))
    if not files:
        print(f"FATAL: no parquet under {base}"); return 1
    have = {f.name for f in pq.ParquetFile(files[0]).schema_arrow}
    cols = [c for c in sp["cols"] if c in have]
    ids, groups, excluded, src_hist, qtype_hist = [], {}, {}, {}, {}
    row = 0
    for f in files:
        t = pq.read_table(f, columns=cols).to_pydict()
        n = len(next(iter(t.values()))) if t else pq.ParquetFile(f).metadata.num_rows
        for j in range(n):
            pid = f"{ds}-{sp['split']}-{row:06d}"
            row += 1
            meta = (t.get("metadata") or [None] * n)[j] or {}
            src = str(meta.get("source", "")) if isinstance(meta, dict) else ""
            src_hist[src] = src_hist.get(src, 0) + 1
            qt = str((t.get("question_type") or [None] * n)[j])
            qtype_hist[qt] = qtype_hist.get(qt, 0) + 1
            if sp.get("exclude_source") and src.lower() in {x.lower() for x in sp["exclude_source"]}:
                excluded[src] = excluded.get(src, 0) + 1
                continue
            ids.append(pid)
            if sp.get("group_col"):
                groups[pid] = str(t[sp["group_col"]][j])
    rep = {"script_version": SCRIPT_VERSION, "dataset": ds, "split": sp["split"],
           "files": [os.path.basename(f) for f in files], "n_rows": row,
           "n_ids": len(ids), "excluded_by_source": excluded,
           "n_groups": len(set(groups.values())) if groups else None,
           "source_hist": dict(sorted(src_hist.items(), key=lambda kv: -kv[1])[:40]),
           "question_type_hist": qtype_hist, "ids": ids, "groups": groups}
    json.dump(rep, open(out, "w"))
    print(f"run={SCRIPT_VERSION} dataset={ds} rows={row} ids={len(ids)} "
          f"excluded={sum(excluded.values())} groups={rep['n_groups']} qtypes={qtype_hist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
