"""Stored-response cache: schema, validation, and loading into per-prompt success counts."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REQUIRED = ("prompt_id", "model", "split", "n_draws", "n_correct")


def _iter_records(path: Path):
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            yield ln, json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{ln}: not valid JSON ({e})")


def load_cache(cache_dir, *, split=None, condition=None) -> list[dict]:
    """Read every .jsonl under cache_dir, validating each record."""
    files = sorted(Path(cache_dir).rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no .jsonl rollout files under {cache_dir}")
    recs = []
    for f in files:
        for ln, r in _iter_records(f):
            miss = [k for k in REQUIRED if k not in r]
            if miss:
                raise ValueError(f"{f}:{ln}: missing keys {miss}")
            d, c = int(r["n_draws"]), int(r["n_correct"])
            if d <= 0:
                raise ValueError(f"{f}:{ln}: n_draws must be positive, got {d}")
            if not 0 <= c <= d:
                raise ValueError(
                    f"{f}:{ln}: n_correct={c} outside [0, n_draws={d}]. A count above "
                    f"the budget means the verifier or the writer is double counting."
                )
            r["n_draws"], r["n_correct"] = d, c
            recs.append(r)
    if split is not None:
        recs = [r for r in recs if r["split"] == split]
    if condition is not None:
        recs = [r for r in recs if r.get("condition", "C0") == condition]
    return recs


def validate(cache_dir, splits_manifest=None, *, min_draws=16) -> dict:
    """Structural and cross-file checks. Returns a summary; raises on anything fatal."""
    recs = load_cache(cache_dir)
    by = defaultdict(dict)                        # (model, split, cond) -> {pid: rec}
    for r in recs:
        key = (r["model"], r["split"], r.get("condition", "C0"))
        pid = r["prompt_id"]
        if pid in by[key]:
            raise ValueError(
                f"duplicate record for prompt {pid} under {key}. Two rows for one "
                f"(model, split, prompt) would double the effective budget."
            )
        by[key][pid] = r

    models = sorted({m for m, _, _ in by})
    problems = []
    # every model must cover the same prompt set within a (split, condition)
    for split, cond in sorted({(s, c) for _, s, c in by}):
        sets = {m: set(by[(m, split, cond)]) for m in models if (m, split, cond) in by}
        if len(sets) < 2:
            continue
        union = set().union(*sets.values())
        for m, s in sets.items():
            if s != union:
                problems.append(
                    f"model '{m}' is missing {len(union - s)} prompts in "
                    f"({split}, {cond}); per-prompt comparisons require a common set"
                )
    low = [(k, p) for k, d in by.items() for p, r in d.items() if r["n_draws"] < min_draws]
    if low:
        problems.append(
            f"{len(low)} records have fewer than {min_draws} draws; Cover@tau at low "
            f"tau is not readable at that budget"
        )
    if splits_manifest is not None:
        declared = {p for v in splits_manifest["splits"].values() for p in v}
        for (m, split, cond), d in by.items():
            stray = set(d) - set(splits_manifest["splits"].get(split, []))
            if stray:
                problems.append(
                    f"{len(stray)} prompts recorded under split '{split}' for model "
                    f"'{m}' are not in that split of the manifest"
                )
            unknown = set(d) - declared
            if unknown:
                problems.append(f"{len(unknown)} prompts for '{m}' are in no split at all")
    if problems:
        raise AssertionError("rollout cache is not usable:\n  - " + "\n  - ".join(problems))
    return {
        "n_records": len(recs),
        "models": models,
        "splits": sorted({s for _, s, _ in by}),
        "conditions": sorted({c for _, _, c in by}),
        "prompts_per_cell": {f"{m}|{s}|{c}": len(d) for (m, s, c), d in sorted(by.items())},
        "total_draws": sum(r["n_draws"] for r in recs),
    }


def rate_matrix(cache_dir, *, split, models, condition="C0", estimator="posterior"):
    """Return (prompt_ids, succ, draws) aligned across `models`.

    `succ` and `draws` are integer arrays of shape (len(models), n_prompts), which is
    exactly what csm_metrics.posterior_rates expects. Raw MLE rates are s/d; the
    shrunk estimator lives in csm_metrics so this module stays I/O only.
    """
    recs = load_cache(cache_dir, split=split, condition=condition)
    idx = defaultdict(dict)
    for r in recs:
        idx[r["model"]][r["prompt_id"]] = r
    for m in models:
        if m not in idx:
            raise KeyError(f"no rollouts for model '{m}' in split '{split}' "
                           f"condition '{condition}'; have {sorted(idx)}")
    common = sorted(set.intersection(*(set(idx[m]) for m in models)))
    if not common:
        raise ValueError("models share no prompts; cannot form a rate matrix")
    succ = np.array([[idx[m][p]["n_correct"] for p in common] for m in models], dtype=int)
    draws = np.array([[idx[m][p]["n_draws"] for p in common] for m in models], dtype=int)
    return common, succ, draws


def generated_tokens(cache_dir, *, split, models, condition="C0") -> dict:
    """Per-model generated-token totals, for the matched-budget accounting E2 needs."""
    recs = load_cache(cache_dir, split=split, condition=condition)
    tot = defaultdict(int)
    missing = 0
    for r in recs:
        if r["model"] in models:
            if "gen_tokens" in r:
                tot[r["model"]] += int(r["gen_tokens"])
            else:
                missing += 1
    if missing:
        print(f"  note: {missing} records carry no gen_tokens; matched-token claims "
              f"cannot be made from this cache", file=sys.stderr)
    return dict(tot)


def write_synthetic_cache(out_dir, *, models=("base", "seed1", "seed2", "seed3"),
                          n_prompts=400, k=32, seed=0, splits=None):
    """SYNTHETIC cache for testing the pipeline end to end. Files are prefixed SYNTH_
    so they can never be mistaken for real runs."""
    rng = np.random.default_rng(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ids = [f"p{i:05d}" for i in range(n_prompts)]
    if splits is None:
        splits = {"val": ids[: n_prompts // 2], "test": ids[n_prompts // 2:]}
    owner = {p: rng.integers(1, len(models)) for p in ids}      # specialist-ish structure
    for mi, m in enumerate(models):
        rows = []
        for split, pids in splits.items():
            for p in pids:
                if m == "base":
                    rate = rng.beta(0.4, 8.0)
                elif owner[p] == mi:
                    rate = rng.uniform(0.30, 0.95)
                else:
                    rate = rng.beta(0.5, 12.0)
                rows.append({"prompt_id": p, "model": m, "split": split,
                             "n_draws": k, "n_correct": int(rng.binomial(k, rate)),
                             "condition": "C0", "gen_tokens": int(k * rng.integers(180, 400))})
        (out / f"SYNTH_{m}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
    return sorted(out.glob("SYNTH_*.jsonl"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="rollout cache tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--cache", required=True)
    v.add_argument("--splits", default=None)
    s = sub.add_parser("selftest")
    s.add_argument("--dir", default="synth_cache")
    a = ap.parse_args(argv)

    if a.cmd == "validate":
        man = None
        if a.splits:
            from csm_splits import load_splits
            man = load_splits(a.splits)
        info = validate(a.cache, man)
        print(f"cache OK: {info['n_records']} records, models={info['models']}, "
              f"splits={info['splits']}, conditions={info['conditions']}, "
              f"total draws={info['total_draws']}")
        return 0

    files = write_synthetic_cache(a.dir)
    print(f"wrote {len(files)} SYNTH_ files to {a.dir}/")
    info = validate(a.dir)
    print(f"validate: {info['n_records']} records, models={info['models']}")
    pids, succ, draws = rate_matrix(a.dir, split="val", models=["seed1", "seed2", "seed3"])
    print(f"rate_matrix(val): {succ.shape[0]} models x {succ.shape[1]} prompts, "
          f"mean rate {float((succ/draws).mean()):.4f}")
    # prove the guards fire
    bad = Path(a.dir) / "SYNTH_bad.jsonl"
    bad.write_text(json.dumps({"prompt_id": "p00000", "model": "seed1", "split": "val",
                               "n_draws": 8, "n_correct": 9}) + "\n")
    try:
        validate(a.dir); print("FAIL  n_correct > n_draws was not caught"); return 1
    except (ValueError, AssertionError):
        print("PASS  n_correct > n_draws is refused")
    bad.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
