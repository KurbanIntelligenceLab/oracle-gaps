"""Merge sampled shards into the stored-response cache."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_VERSION = "csm_cache_build/1.0"

REQUIRED_COLS = ("prompt_id", "model", "split", "n_draws", "n_correct")


def build(merged_csv: str, out_dir: str, *, dataset_col: str = "dataset") -> dict:
    src = Path(merged_csv)
    if not src.exists():
        raise FileNotFoundError(src)

    with src.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{src} has no rows")

    missing = [c for c in REQUIRED_COLS if c not in rows[0]]
    if missing:
        raise ValueError(f"{src} is missing required columns {missing}; "
                         f"found {sorted(rows[0])}")

    seen: set[tuple[str, str, str, str]] = set()
    by_file: dict[tuple[str, str], list[dict]] = defaultdict(list)
    n_bad = 0

    for i, r in enumerate(rows, 2):          # 2 = first data line
        pid = r["prompt_id"].strip()
        model = r["model"].strip()
        split = r["split"].strip()
        cond = (r.get("condition") or "C0").strip() or "C0"
        try:
            d = int(r["n_draws"])
            c = int(r["n_correct"])
        except (TypeError, ValueError):
            raise ValueError(f"{src}:{i}: n_draws/n_correct not integers: {r}")
        if d <= 0:
            raise ValueError(f"{src}:{i}: n_draws must be positive, got {d}")
        if not 0 <= c <= d:
            raise ValueError(
                f"{src}:{i}: n_correct={c} outside [0,{d}]. A count above the "
                f"budget means the verifier or the writer is double counting.")

        key = (model, split, cond, pid)
        if key in seen:
            raise ValueError(
                f"{src}:{i}: duplicate record for prompt {pid} under "
                f"({model},{split},{cond}). Two rows for one (model,split,prompt) "
                f"would double the effective budget.")
        seen.add(key)

        rec = {"prompt_id": pid, "model": model, "split": split,
               "n_draws": d, "n_correct": c}
        if cond != "C0":
            rec["condition"] = cond
        if r.get("gen_tokens"):
            try:
                rec["gen_tokens"] = int(r["gen_tokens"])
            except (TypeError, ValueError):
                n_bad += 1
        if r.get(dataset_col):
            rec["dataset"] = r[dataset_col].strip()
        by_file[(model, split)].append(rec)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for (model, split), recs in sorted(by_file.items()):
        p = out / f"{model}__{split}.jsonl"
        with p.open("w") as f:
            for rec in sorted(recs, key=lambda r: r["prompt_id"]):
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        written[str(p)] = len(recs)

    summary = {
        "script_version": SCRIPT_VERSION,
        "source": str(src),
        "out_dir": str(out),
        "n_records": len(rows),
        "files": written,
        "models": sorted({m for m, _ in by_file}),
        "splits": sorted({s for _, s in by_file}),
        "unparsable_gen_tokens": n_bad,
    }
    return summary


def selftest() -> int:
    import tempfile
    ok = fail = 0

    def check(cond, label):
        nonlocal ok, fail
        if cond:
            ok += 1
        else:
            fail += 1
            print(f"FAIL  {label}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        good = td / "good.csv"
        good.write_text(
            "prompt_id,model,split,dataset,n_draws,n_correct,gen_tokens\n"
            "p1,base,val,chartqa,16,3,1000\n"
            "p2,base,val,chartqa,16,0,900\n"
            "p1,seed1,val,chartqa,16,7,1100\n")
        s = build(str(good), str(td / "cache"))
        check(s["n_records"] == 3, "row count")
        check(sorted(s["models"]) == ["base", "seed1"], "models split by file")
        check(len(s["files"]) == 2, "one file per (model,split)")

        # the built cache must satisfy the real validator
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import csm_rollouts
        recs = csm_rollouts.load_cache(str(td / "cache"))
        check(len(recs) == 3, "validator loads what we wrote")

        dup = td / "dup.csv"
        dup.write_text(
            "prompt_id,model,split,n_draws,n_correct\n"
            "p1,base,val,16,3\n"
            "p1,base,val,16,4\n")
        try:
            build(str(dup), str(td / "c2"))
            check(False, "duplicate rows must be refused")
        except ValueError as e:
            check("duplicate" in str(e), "duplicate rows refused")

        bad = td / "bad.csv"
        bad.write_text(
            "prompt_id,model,split,n_draws,n_correct\n"
            "p1,base,val,16,17\n")
        try:
            build(str(bad), str(td / "c3"))
            check(False, "n_correct > n_draws must be refused")
        except ValueError as e:
            check("outside" in str(e), "n_correct > n_draws refused")

        miss = td / "miss.csv"
        miss.write_text("prompt_id,model\np1,base\n")
        try:
            build(str(miss), str(td / "c4"))
            check(False, "missing columns must be refused")
        except ValueError as e:
            check("missing required columns" in str(e), "missing columns refused")

    print(f"csm_cache_build selftest: {ok} passed, {fail} failed")
    return 1 if fail else 0


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "selftest":
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--merged", required=True, help="merged rollout CSV")
    ap.add_argument("--out", required=True, help="output cache directory")
    a = ap.parse_args(argv)
    s = build(a.merged, a.out)
    print(json.dumps(s, indent=1))
    print(f"cache_built records={s['n_records']} models={','.join(s['models'])} "
          f"splits={','.join(s['splits'])} out={s['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
