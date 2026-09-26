"""Apply the fixed second-dataset rules to base-model validation responses."""
from __future__ import annotations
import pathlib as _pl, sys as _sys
_REPO_ROOT = next(p for p in _pl.Path(__file__).resolve().parents if (p / "lib").is_dir())
_sys.path.insert(0, str(_REPO_ROOT / "lib"))  # shared modules
import argparse, hashlib, json, sys
from pathlib import Path

SCRIPT_VERSION = "csm_ds2_decide/1.0"
ACCEPT_MIN = 0.50
TIE_POINTS = 0.03
BAND_PRIOR = "[0.05,0.30]"
BAND_SWITCH_POINTS = 0.05
PREFERRED = "mathvista"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="analysis/tau_calibration_ds2.json")
    ap.add_argument("--prereg", default="analysis/preregistration_ds2.json")
    ap.add_argument("--out", default="analysis/ds2_decision.json")
    ap.add_argument("--choice-file", default="analysis/ds2_dataset.txt")
    a = ap.parse_args()
    cal = json.loads(Path(a.calibration).read_text())
    pre = json.loads(Path(a.prereg).read_text())
    pre_hash = hashlib.sha256(json.dumps(pre["declaration"], sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()
    if pre_hash != pre.get("sha256"):
        print("FATAL: preregistration_ds2.json does not match its own hash"); return 1

    table = {}
    for ds, d in cal["datasets"].items():
        nd = float(d["frac_non_degenerate"])
        bands = {b: float(v["frac_of_non_degenerate"]) for b, v in d["bands"].items()}
        table[ds] = {"n_items": d["n_items"], "mean_rate": d["mean_rate"],
                     "frac_zero": d["frac_zero"], "frac_one": d["frac_one"],
                     "frac_non_degenerate": nd, "accepted": nd >= ACCEPT_MIN,
                     "band_capture_of_non_degenerate": bands}
    accepted = [ds for ds, t in table.items() if t["accepted"]]
    chosen, why = None, ""
    if accepted:
        best = max(accepted, key=lambda ds: table[ds]["frac_non_degenerate"])
        if PREFERRED in accepted and best != PREFERRED and \
           table[best]["frac_non_degenerate"] - table[PREFERRED]["frac_non_degenerate"] <= TIE_POINTS:
            chosen, why = PREFERRED, f"tie within {TIE_POINTS:.2f} of {best}; preferred for the domain shift"
        else:
            chosen, why = best, "largest non-degenerate fraction among accepted candidates"
    band, band_why = BAND_PRIOR, "prior band kept"
    if chosen:
        caps = table[chosen]["band_capture_of_non_degenerate"]
        prior = caps.get(BAND_PRIOR, 0.0)
        alt = max(caps, key=caps.get)
        if alt != BAND_PRIOR and caps[alt] - prior >= BAND_SWITCH_POINTS:
            band, band_why = alt, f"{alt} captures {caps[alt]:.3f} of non-degenerate mass vs {prior:.3f} for the prior, exceeding the {BAND_SWITCH_POINTS:.2f} switch rule"
    out = {"script_version": SCRIPT_VERSION, "rules": {"accept_min": ACCEPT_MIN, "tie_points": TIE_POINTS,
           "band_prior": BAND_PRIOR, "band_switch_points": BAND_SWITCH_POINTS, "preferred": PREFERRED},
           "prereg_sha256": pre_hash, "candidates": table, "accepted": accepted,
           "chosen": chosen, "why": why, "band": band, "band_why": band_why}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for ds, t in table.items():
        print(f"{ds}: n={t['n_items']} mean_rate={t['mean_rate']:.3f} zero={t['frac_zero']:.3f} "
              f"one={t['frac_one']:.3f} non_degenerate={t['frac_non_degenerate']:.3f} "
              f"accepted={t['accepted']} band_prior_capture={t['band_capture_of_non_degenerate'].get(BAND_PRIOR, float('nan')):.3f}")
    if chosen:
        Path(a.choice_file).write_text(chosen + "\n")
        print(f"DECISION: dataset={chosen} ({why}); band={band} ({band_why}); wrote {a.choice_file}")
        return 0
    print("DECISION: NO CANDIDATE ACCEPTED -- the second-dataset study does not proceed on these data")
    return 2


if __name__ == "__main__":
    sys.exit(main())
