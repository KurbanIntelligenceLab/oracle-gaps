"""Hashed, disjoint train/validation/test prompt splits and the checks that enforce them."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FRACTIONS = {"train": 0.40, "teacher": 0.25, "val": 0.15, "test": 0.20}
NAMES = list(FRACTIONS)


def _digest(items) -> str:
    h = hashlib.sha256()
    for x in items:
        h.update(str(x).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def make_splits(ids, seed: int = 0, fractions=None, groups=None) -> dict:
    """Deterministic disjoint partition. `ids` must be unique prompt identifiers.

    `groups` (optional) maps prompt_id -> group key. Every prompt of one group
    lands in the SAME split. This exists for datasets that carry several
    versions of one underlying problem (MathVerse has five per problem): a
    row-level partition would put versions of one problem on both sides of the
    train/test line, which is the contamination this module exists to prevent.
    With groups=None the partition is exactly the original row-level one.
    """
    fr = dict(fractions or FRACTIONS)
    ids = list(dict.fromkeys(str(i) for i in ids))          # de-duplicate, keep order
    if len(ids) < 20:
        raise ValueError("need at least 20 prompts to split meaningfully")
    if abs(sum(fr.values()) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {sum(fr.values())}")

    out, start = {}, 0
    if groups is None:
        # deterministic shuffle that does not depend on the numpy version
        order = sorted(ids, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
        for k in NAMES[:-1]:
            n = int(round(fr[k] * len(order)))
            out[k] = sorted(order[start:start + n])
            start += n
        out[NAMES[-1]] = sorted(order[start:])               # remainder, no prompt lost
        n_groups = None
    else:
        missing = [i for i in ids if i not in groups]
        if missing:
            raise ValueError(f"{len(missing)} prompts have no group, e.g. {missing[:3]}")
        members: dict[str, list[str]] = {}
        for i in ids:
            members.setdefault(str(groups[i]), []).append(i)
        gorder = sorted(members, key=lambda g: hashlib.sha256(f"{seed}:g:{g}".encode()).hexdigest())
        targets = [int(round(fr[k] * len(ids))) for k in NAMES[:-1]]
        buckets = {k: [] for k in NAMES}
        ki = 0
        for g in gorder:
            if ki < len(targets) and len(buckets[NAMES[ki]]) >= targets[ki]:
                ki += 1
            buckets[NAMES[min(ki, len(NAMES) - 1)]].extend(members[g])
        out = {k: sorted(v) for k, v in buckets.items()}
        n_groups = len(members)

    manifest = {
        "seed": seed,
        "fractions": fr,
        "n_total": len(ids),
        "splits": out,
        "sizes": {k: len(v) for k, v in out.items()},
        "hashes": {k: _digest(v) for k, v in out.items()},
        "all_hash": _digest(sorted(ids)),
    }
    if n_groups is not None:
        manifest["grouped"] = True
        manifest["n_groups"] = n_groups
        manifest["group_hash"] = _digest(sorted(f"{i}\t{groups[i]}" for i in ids))
    _assert_valid(manifest)
    return manifest


def _assert_valid(m: dict) -> None:
    sp = m["splits"]
    missing = [k for k in NAMES if k not in sp]
    if missing:
        raise AssertionError(f"manifest is missing splits: {missing}")
    for k in NAMES:
        if len(set(sp[k])) != len(sp[k]):
            raise AssertionError(f"split '{k}' contains duplicate prompt ids")
        if not sp[k]:
            raise AssertionError(f"split '{k}' is empty")
    # the assertion the whole paper rests on
    for i, a in enumerate(NAMES):
        for b in NAMES[i + 1:]:
            inter = set(sp[a]) & set(sp[b])
            if inter:
                raise AssertionError(
                    f"CONTAMINATION: {len(inter)} prompts appear in both '{a}' and "
                    f"'{b}', e.g. {sorted(inter)[:5]}. Any coverage number computed "
                    f"from these splits is invalid. Rebuild the splits."
                )
    total = sum(len(sp[k]) for k in NAMES)
    if total != m["n_total"]:
        raise AssertionError(f"splits cover {total} prompts but n_total={m['n_total']}")
    for k in NAMES:
        if _digest(sp[k]) != m["hashes"][k]:
            raise AssertionError(
                f"split '{k}' does not match its recorded hash; the file was edited "
                f"after creation. Rebuild rather than patch."
            )


def load_splits(path="splits.json") -> dict:
    """Load and re-police. Every downstream script must go through this."""
    m = json.loads(Path(path).read_text())
    _assert_valid(m)
    return m


def assert_reportable(prompt_ids, manifest, *, arm: str) -> None:
    """Call before computing any reported number on `prompt_ids`.

    Fails if the prompts touch the teacher pool or the training set, which is the
    error the paper cannot survive.
    """
    ids = set(str(i) for i in prompt_ids)
    for forbidden in ("teacher", "train"):
        bad = ids & set(manifest["splits"][forbidden])
        if bad:
            raise AssertionError(
                f"arm '{arm}' is about to report on {len(bad)} prompts that are in the "
                f"'{forbidden}' split, e.g. {sorted(bad)[:5]}. This is the "
                f"train-test contamination. Stop and fix the arm."
            )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make");  mk.add_argument("--ids", required=True)
    mk.add_argument("--out", default="splits.json"); mk.add_argument("--seed", type=int, default=0)
    mk.add_argument("--groups", default=None,
                    help="TSV 'prompt_id<TAB>group'; whole groups land in one split")
    ck = sub.add_parser("check"); ck.add_argument("--splits", default="splits.json")
    dm = sub.add_parser("selftest")
    a = ap.parse_args(argv)

    if a.cmd == "make":
        ids = [l.strip() for l in Path(a.ids).read_text().splitlines() if l.strip()]
        groups = None
        if a.groups:
            groups = dict(l.rstrip("\n").split("\t", 1)
                          for l in Path(a.groups).read_text().splitlines() if l.strip())
        m = make_splits(ids, seed=a.seed, groups=groups)
        if groups is not None:
            print(f"grouped: {m['n_groups']} groups, group_hash={m['group_hash']}")
        Path(a.out).write_text(json.dumps(m, indent=2))
        print(f"wrote {a.out}: " + "  ".join(f"{k}={m['sizes'][k]}" for k in NAMES))
        print("hashes: " + "  ".join(f"{k}={m['hashes'][k]}" for k in NAMES))
        return 0

    if a.cmd == "check":
        m = load_splits(a.splits)
        print("splits valid and disjoint: " + "  ".join(f"{k}={m['sizes'][k]}" for k in NAMES))
        return 0

    # selftest: prove the guard actually fires
    m = make_splits([f"p{i:04d}" for i in range(500)], seed=0)
    print("built:", {k: m["sizes"][k] for k in NAMES})
    try:
        assert_reportable(m["splits"]["test"], m, arm="E5")
        print("PASS  reporting on test is allowed")
    except AssertionError as e:
        print("FAIL  test split rejected:", e); return 1
    try:
        assert_reportable(m["splits"]["teacher"][:3], m, arm="E11")
        print("FAIL  contamination was NOT caught"); return 1
    except AssertionError:
        print("PASS  reporting on teacher prompts is refused")
    bad = json.loads(json.dumps(m))
    bad["splits"]["val"].append(bad["splits"]["test"][0])
    try:
        _assert_valid(bad); print("FAIL  overlap was NOT caught"); return 1
    except AssertionError:
        print("PASS  val/test overlap is refused")
    bad2 = json.loads(json.dumps(m)); bad2["splits"]["train"].pop()
    try:
        _assert_valid(bad2); print("FAIL  post-hoc edit was NOT caught"); return 1
    except AssertionError:
        print("PASS  hash mismatch after editing the file is refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
