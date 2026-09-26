"""Check that training runs share base checkpoint, prompts and configuration apart from the seed."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, hashlib, json, sys
from pathlib import Path

SEED_KEYS = ("seed", "data_order_seed", "init_seed", "rollout_seed", "sampling_seed")


def _hash_path_or_str(v) -> str:
    p = Path(str(v))
    h = hashlib.sha256()
    if p.exists() and p.is_file():
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return "file:" + h.hexdigest()[:16]
    if p.exists() and p.is_dir():
        for q in sorted(p.rglob("*")):
            if q.is_file():
                h.update(q.name.encode()); h.update(str(q.stat().st_size).encode())
        return "dir:" + h.hexdigest()[:16]
    h.update(str(v).encode())
    return "str:" + h.hexdigest()[:16]


def _config_hash(cfg: dict) -> str:
    stripped = {k: v for k, v in cfg.items() if k not in SEED_KEYS}
    return "cfg:" + hashlib.sha256(
        json.dumps(stripped, sort_keys=True).encode()).hexdigest()[:16]


def audit(parents: list[dict]) -> dict:
    if len(parents) < 2:
        raise ValueError("E0 needs at least two parents")
    rows = []
    for p in parents:
        for k in ("name", "base_checkpoint", "data_manifest", "config"):
            if k not in p:
                raise ValueError(f"parent record missing '{k}': {p.get('name', p)}")
        seeds = {k: p["config"].get(k) for k in SEED_KEYS if k in p["config"]}
        if not seeds:
            raise ValueError(f"parent '{p['name']}' declares no seed field; one of "
                             f"{SEED_KEYS} is required")
        rows.append({"name": p["name"],
                     "base": _hash_path_or_str(p["base_checkpoint"]),
                     "data": _hash_path_or_str(p["data_manifest"]),
                     "config_wo_seed": _config_hash(p["config"]),
                     "seeds": seeds})
    problems = []
    for field, label in (("base", "base checkpoint"), ("data", "data manifest"),
                         ("config_wo_seed", "config with seeds removed")):
        vals = {r[field] for r in rows}
        if len(vals) != 1:
            problems.append(
                f"parents disagree on the {label}: "
                + ", ".join(f"{r['name']}={r[field]}" for r in rows)
                + ". The pool is not same-base seed-only, so no downstream claim holds.")
    sig = [tuple(sorted(r["seeds"].items())) for r in rows]
    if len(set(sig)) != len(sig):
        dup = [rows[i]["name"] for i in range(len(sig)) if sig.count(sig[i]) > 1]
        problems.append(f"parents share an identical seed configuration ({dup}); "
                        f"these are not independent replicates")
    ok = not problems
    return {"passed": ok, "parents": rows, "problems": problems}


def _print(res: dict) -> int:
    print(f"{'parent':10s} {'base':22s} {'data':22s} {'config(no seed)':22s} seeds")
    for r in res["parents"]:
        print(f"{r['name']:10s} {r['base']:22s} {r['data']:22s} "
              f"{r['config_wo_seed']:22s} {r['seeds']}")
    if res["passed"]:
        print("\nE0 PASSED: identical base, data and config; seeds differ. "
              "'same base, seed only' is supported by the recorded evidence.")
        return 0
    print("\nE0 FAILED. Hard blocker:")
    for p in res["problems"]:
        print("  - " + p)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("audit"); a1.add_argument("--parents", nargs="+", required=True)
    a1.add_argument("--out", default="ancestry.json")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)

    if a.cmd == "audit":
        parents = [json.loads(Path(p).read_text()) for p in a.parents]
        res = audit(parents)
        Path(a.out).write_text(json.dumps(res, indent=2))
        return _print(res)

    base = {"base_checkpoint": "BASE-SNAPSHOT-A", "data_manifest": "DATA-MANIFEST-A",
            "config": {"lr": 1e-6, "steps": 500, "trainable_modules": ["llm"]}}
    good = [dict(base, name=f"seed{i}", config=dict(base["config"], seed=i,
            data_order_seed=i, init_seed=i)) for i in (1, 2, 3)]
    print("--- three genuine seed replicates ---")
    if _print(audit(good)) != 0:
        return 1
    print("\n--- one parent trained on different data (must FAIL) ---")
    bad = [dict(g) for g in good]; bad[2] = dict(bad[2], data_manifest="DATA-MANIFEST-B")
    if _print(audit(bad)) == 0:
        print("SELFTEST FAILED: a data mismatch was not caught"); return 1
    print("\n--- two parents sharing a seed (must FAIL) ---")
    dup = [dict(g) for g in good]
    dup[2] = dict(dup[2], config=dict(dup[0]["config"]))
    if _print(audit(dup)) == 0:
        print("SELFTEST FAILED: a duplicate seed was not caught"); return 1
    print("\nselftest complete: E0 passes genuine replicates and refuses both failure modes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
