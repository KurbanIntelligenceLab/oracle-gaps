"""Turn enumerated prompt identifiers into hashed splits and probe files."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csm_splits as S  # noqa: E402


def probe(ids, n, seed):
    order = sorted(ids, key=lambda i: hashlib.sha256(f"probe:{seed}:{i}".encode()).hexdigest())
    return sorted(order[:n])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--probe", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="analysis/splits")
    a = ap.parse_args()
    r = json.load(open(a.shard))
    ds, ids, groups = r["dataset"], r["ids"], r.get("groups") or None
    os.makedirs(a.outdir, exist_ok=True)
    p = lambda name: os.path.join(a.outdir, f"{ds}_{name}")
    open(p("ids.txt"), "w").write("\n".join(ids) + "\n")
    if groups:
        open(p("groups.tsv"), "w").write("".join(f"{i}\t{groups[i]}\n" for i in ids))
    m = S.make_splits(ids, seed=a.seed, groups=groups)
    m["source"] = {"shard": os.path.basename(a.shard), "script_version": r.get("script_version"),
                   "n_rows": r.get("n_rows"), "excluded_by_source": r.get("excluded_by_source")}
    json.dump(m, open(p("splits.json"), "w"), indent=2)
    S.load_splits(p("splits.json"))                      # re-police what was written
    open(p("train_rlvr.txt"), "w").write("\n".join(m["splits"]["train"]) + "\n")
    for split in ("val", "test"):
        ids_s = probe(m["splits"][split], a.probe, a.seed)
        S.assert_reportable(ids_s, m, arm=f"{ds}-{split}-probe")
        open(p(f"{split}_probe.txt"), "w").write("\n".join(ids_s) + "\n")
    print(f"{ds}: ids={len(ids)} " + " ".join(f"{k}={m['sizes'][k]}" for k in S.NAMES)
          + (f" groups={m['n_groups']}" if groups else "")
          + f" val_probe={min(a.probe, m['sizes']['val'])} test_probe={min(a.probe, m['sizes']['test'])}")
    print("hashes: " + "  ".join(f"{k}={m['hashes'][k]}" for k in S.NAMES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
