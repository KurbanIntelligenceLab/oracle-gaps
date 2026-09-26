#!/usr/bin/env python3
"""Count the stored verdicts that a checker change flips."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import glob, json, os, re, sys, collections

import csm_verify as V                                            # noqa: E402

SCRIPT_VERSION = "verifier_impact/1.2"

# The patch under test, applied to the extracted answer before the real
# normaliser sees it. Kept here rather than in csm_verify so this job measures
# the delta without changing the module it is measuring.
_DEG = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|\\circ\b|\\degree\b|\\deg\b")
_SQUARED_UNIT = re.compile(
    r"\b(cm|mm|km|m|in|ft|units?|sq)\b\s*\^\s*\{?\s*([23])\s*\}?", re.IGNORECASE)
_MIXED = re.compile(r"(?<![\d.])(\d+)\s*\\[dt]?frac\s*\{(\d+)\}\s*\{(\d+)\}")


def patched(s: str) -> str:
    if s is None:
        return ""
    t = str(s)
    t = _MIXED.sub(lambda m: f"({m.group(1)}+{m.group(2)}/{m.group(3)})", t)
    t = _SQUARED_UNIT.sub("", t)      # drop the unit AND its exponent together
    t = _DEG.sub("", t)
    return t


def main() -> int:
    out = os.environ.get("OUT", "verifier_impact.csv")
    root = os.path.join(os.environ.get("DATA_DIR", ""), "CSM", "rollouts_raw")
    dirs = sorted(glob.glob(os.path.join(root, "*__geometry3k__*")))

    pat = collections.Counter()
    flips = collections.Counter()
    goldfmt = collections.Counter()      # what shapes do the GOLDS actually take?
    gold_bad = []                        # golds the verifier cannot parse at all
    seen_gold = set()
    n = 0          # counts ROLLOUTS, not prompt records
    examples = []
    for d in dirs:
        arm_split = os.path.basename(d)
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            for line in open(f):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                gold = r.get("gold")
                # Each record holds ONE prompt with a LIST of completions, one
                # per draw. Reading a singular "completion" key silently yields
                # nothing, which is how the first version of this job reported
                # zero patterns over what looked like 3600 rollouts.
                # Audit the GOLD too. A gold the parser mis-reads corrupts the
                # score for every arm at once, which is worse than any
                # prediction-side defect. The mixed-number form a\frac{b}{c} is
                # the specific worry: it parses as a*(b/c), not a+(b/c).
                gkey = (r.get("dataset"), str(gold))
                if gkey not in seen_gold:
                    seen_gold.add(gkey)
                    g = str(gold)
                    if _MIXED.search(g):
                        goldfmt["mixed_number"] += 1
                        gold_bad.append(("mixed", g))
                    elif "frac" in g:
                        goldfmt["latex_frac"] += 1
                    elif "sqrt" in g:
                        goldfmt["latex_sqrt"] += 1
                    elif re.fullmatch(r"-?\d+", g):
                        goldfmt["integer"] += 1
                    elif re.fullmatch(r"-?\d*\.\d+", g):
                        goldfmt["decimal"] += 1
                    else:
                        goldfmt["other"] += 1
                        if len(gold_bad) < 40:
                            gold_bad.append(("other", g))
                    if V.to_number(V.normalize_text(g)) is None:
                        goldfmt["UNPARSEABLE"] += 1
                        if len(gold_bad) < 40:
                            gold_bad.append(("unparseable", g))
                comps = r.get("completions")
                if isinstance(comps, str):
                    comps = [comps]
                for comp in (comps or []):
                    ans = V.extract_answer(comp)
                    n += 1
                    if _DEG.search(ans):
                        pat["degree_mark"] += 1
                    if _SQUARED_UNIT.search(ans):
                        pat["unit_exponent"] += 1
                    if _MIXED.search(ans):
                        pat["mixed_number"] += 1
                    now = V.verify(ans, gold, "geometry3k")
                    fixed = V.verify(patched(ans), gold, "geometry3k")
                    if now != fixed:
                        key = ("false_negative_recovered" if fixed
                               else "newly_rejected")
                        flips[key] += 1
                        flips[f"{key}::{arm_split}"] += 1
                        if len(examples) < 25:
                            examples.append(
                                (arm_split, str(gold)[:20], ans[:60], now, fixed))

    with open(out, "w") as fh:
        fh.write("metric,value\n")
        fh.write(f"n_rollouts,{n}\n")
        for k, v in sorted(pat.items()):
            fh.write(f"pattern_{k},{v}\n")
        for k, v in sorted(flips.items()):
            fh.write(f"flip_{k},{v}\n")
        for k, v in sorted(goldfmt.items()):
            fh.write(f"goldfmt_{k},{v}\n")
        fh.write(f"distinct_golds,{len(seen_gold)}\n")
        rate = (flips['false_negative_recovered'] / n) if n else 0.0
        fh.write(f"recovered_rate,{rate:.6f}\n")

    print(f"run={SCRIPT_VERSION} rollouts={n} dirs={len(dirs)}")
    print("patterns: " + " ".join(f"{k}={v}" for k, v in sorted(pat.items())))
    print("flips: " + " ".join(f"{k}={v}" for k, v in sorted(flips.items())
                               if "::" not in k))
    if n:
        print(f"recovered_rate={flips['false_negative_recovered']/n:.4f} "
              f"newly_rejected={flips['newly_rejected']}")
    print("gold formats (distinct golds=%d): " % len(seen_gold)
          + " ".join(f"{k}={v}" for k, v in sorted(goldfmt.items())))
    for kind, g in gold_bad[:12]:
        print(f"  gold[{kind}] {g!r}")
    for e in examples[:12]:
        print(f"  eg {e[0]} gold={e[1]!r} ans={e[2]!r} now={e[3]} fixed={e[4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
