"""Write and verify the second-dataset selection rules before any test scoring."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, hashlib, json, sys
from pathlib import Path

SCRIPT_VERSION = "csm_prereg_ds2/1.0"

DECLARATION = {
    "declared_on": "2026-09-06",
    "declared_before_touching_test_data": True,
    "purpose": "External validity of the first study: same backbone, same recipe, same "
               "estimands, on a dataset whose distribution differs materially from Geometry3K.",
    "backbone": {"repo_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                 "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5", "unchanged": True},

    "candidates": {
        "mathvista": {
            "split_used": "testmini", "reason_public_answers": "only testmini carries answers",
            "exclusion": "rows whose metadata.source is Geometry3K (62 of 1000) or GEOS "
                         "(22 of 1000), the geometry lineage the first study drew on: a "
                         "second dataset must not contain the first",
            "grouping": None,
            "why_preferred_a_priori": "about 30% geometry; the rest is charts, tables, "
                                      "synthetic scenes, function plots and textbook figures",
        },
        "mathverse": {
            "split_used": "testmini",
            "exclusion": "rows whose metadata.source is Geometry3K (645 of 3940)",
            "grouping": "problem_index, so the five versions of one problem share a split",
            "why_calibrated_anyway": "same-domain control: 82% geometry, so it tests the "
                                     "recipe but not the domain shift",
        },
    },

    "split_rule": {
        "partition": "csm_splits.make_splits, seed 0, fractions train .40 / teacher .25 / "
                     "val .15 / test .20, hashed and policed as in the first study",
        "probe": "per split, order ids by sha256('probe:0:<id>'), take the first 240 (or the "
                 "whole split if smaller), sort",
        "train_prompts": "min(400, |train|), first ids of the train split in file order",
    },

    "calibration_rule": {
        "job": "P3: base model, k=16, temperature 1.0, 512 new tokens, on the val probe",
        "accept_dataset_if": "non-degenerate fraction (items with 0 < rate < 1) >= 0.50; "
                             "Geometry3K measured 0.600 and ChartQA 0.446 was rejected",
        "band": "the first study's [0.05, 0.30] is the prior and is KEPT unless another "
                "candidate band captures at least 5 absolute points more of the "
                "non-degenerate mass, in which case that band is used and reported as "
                "a change with its reason",
        "candidate_bands": [[0.05, 0.30], [0.10, 0.40], [0.15, 0.50], [0.20, 0.60],
                            [0.30, 0.70], [0.40, 0.80], [0.50, 0.90]],
        "choose_dataset": "the accepted candidate with the larger non-degenerate fraction; "
                          "MathVista preferred on a tie within 3 points because of the "
                          "domain shift",
    },

    "training_rule": {
        "seeds": [1, 2, 3, 4, 5],
        "recipe": "identical to rlvr_train.sh defaults: GRPO, LoRA r=16 alpha=32 on the LLM, "
                  "lr 1e-5, 500 steps, 8 generations, beta 0, max completion 512; hardware "
                  "pinned to bf16-capable nodes (run --uniform)",
        "only_difference_across_seeds": "the seed (csm_ancestry.py verifies)",
    },

    "measurement_rule": {
        "k": 16, "max_new_tokens": 1024, "arms": "base + five seeds on val and test probes",
        "verifier": "csm_verify 2.0 with tolerance 1e-3 for both candidates; multiple-choice "
                    "golds accept the option letter or the option text",
        "estimands": "unchanged: Cover@tau on the tau grid, rAUC on the band, D_N, L, G_multi "
                     "with its base-weight-matched control under both comparator rules, "
                     "M-scaling by offline subsampling, routing e_D fitted on val",
        "specialist_mark": "median slack over the factor-M floor <= 0.05 absolute coverage",
        "smallest_effect_worth_reporting_abs_points": 0.03,
    },

    "reporting_rule": "All outcomes are reported whatever their sign. A bounded null is a "
                      "result. Nothing in this study revises the first study's numbers.",
    "test_split_status": "UNTOUCHED at declaration.",
}


def _canon(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def emit(out: str) -> int:
    dec = dict(DECLARATION)
    hashes = {}
    for ds in dec["candidates"]:
        f = Path(f"analysis/splits/{ds}_splits.json")
        if f.exists():
            m = json.loads(f.read_text())
            hashes[ds] = {"n_total": m["n_total"], "sizes": m["sizes"], "hashes": m["hashes"],
                          "all_hash": m["all_hash"], "grouped": m.get("grouped", False)}
    dec["split_hashes"] = hashes or "splits not yet built at emit time"
    payload = {"script_version": SCRIPT_VERSION, "declaration": dec,
               "sha256": hashlib.sha256(_canon(dec).encode()).hexdigest()}
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"run={SCRIPT_VERSION} wrote={p} sha256={payload['sha256']} "
          f"splits_recorded={list(hashes)}")
    return 0


def verify(file: str) -> int:
    payload = json.loads(Path(file).read_text())
    got = hashlib.sha256(_canon(payload["declaration"]).encode()).hexdigest()
    ok = payload.get("sha256") == got
    print(f"file={file} MATCH={ok}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit"); e.add_argument("--out", default="analysis/preregistration_ds2.json")
    v = sub.add_parser("verify"); v.add_argument("--file", default="analysis/preregistration_ds2.json")
    a = ap.parse_args(argv)
    return emit(a.out) if a.cmd == "emit" else verify(a.file)


if __name__ == "__main__":
    sys.exit(main())
