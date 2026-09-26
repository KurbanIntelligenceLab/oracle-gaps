#!/usr/bin/env python3
"""Loaders for the verdict and rate files used by the offline analyses."""

from __future__ import annotations
import os
import sys
import numpy as np

REQUIRED_VERDICTS = ("verdicts", "base_verdicts", "prompt_ids", "member_ids", "split")
REQUIRED_RATES = ("R", "b", "alpha", "beta", "fit_split", "prompt_ids", "member_ids", "split")


class DataMissing(FileNotFoundError):
    """Raised when an artefact this package cannot invent is absent."""


class DataInconsistent(ValueError):
    """Raised when the artefacts disagree with each other."""


def _require(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise DataMissing(
            f"{path} not found.\n"
            f"  This script needs {what}. It is not shipped with the submission and\n"
            f"  cannot be reconstructed from the manuscript. See DATA_SCHEMA.md for the\n"
            f"  format, produce the file from your rollouts, then re-run."
        )
    return path


def load_verdicts(path: str = "data/verdicts.npz", split: str | None = None):
    """Raw 0/1 verdicts. Needed by the negative control and the split-half debias."""
    d = np.load(_require(path, "the raw per-rollout verdicts"), allow_pickle=False)
    missing = [k for k in REQUIRED_VERDICTS if k not in d]
    if missing:
        raise DataInconsistent(f"{path} is missing keys {missing}; see DATA_SCHEMA.md")

    V = np.asarray(d["verdicts"])
    if V.ndim != 3:
        raise DataInconsistent(f"verdicts must be (n_prompts, M, k); got {V.shape}")
    uniq = set(np.unique(V).tolist())
    if not uniq <= {0, 1}:
        raise DataInconsistent(f"verdicts must be 0/1; found values {sorted(uniq)[:6]}")

    out = {k: np.asarray(d[k]) for k in d.files}
    if split is not None:
        sel = out["split"] == split
        if not sel.any():
            raise DataInconsistent(
                f"split={split!r} selects no prompts; present: {sorted(set(out['split'].tolist()))}"
            )
        for k in ("verdicts", "base_verdicts", "prompt_ids", "split"):
            if k in out:
                out[k] = out[k][sel]
        if "n_gen_tokens" in out:
            out["n_gen_tokens"] = out["n_gen_tokens"][sel]
    return out


def load_rates(path: str = "data/rates.npz", split: str | None = None):
    """Shrunk per-prompt rates, plus the fitted prior."""
    d = np.load(_require(path, "the shrunk per-prompt rate profiles"), allow_pickle=False)
    missing = [k for k in REQUIRED_RATES if k not in d]
    if missing:
        raise DataInconsistent(f"{path} is missing keys {missing}; see DATA_SCHEMA.md")

    R = np.asarray(d["R"], dtype=float)
    b = np.asarray(d["b"], dtype=float)
    if R.ndim != 2:
        raise DataInconsistent(f"R must be (n_prompts, M); got {R.shape}")
    for name, A in (("R", R), ("b", b)):
        if not np.isfinite(A).all():
            raise DataInconsistent(f"{name} contains NaN or inf")
        if A.min() < 0 or A.max() > 1:
            raise DataInconsistent(f"{name} must lie in [0,1]; got [{A.min()}, {A.max()}]")

    out = {k: np.asarray(d[k]) for k in d.files}
    if split is not None:
        sel = out["split"] == split
        if not sel.any():
            raise DataInconsistent(
                f"split={split!r} selects no prompts; present: {sorted(set(out['split'].tolist()))}"
            )
        for k in ("R", "b", "prompt_ids", "split"):
            out[k] = out[k][sel]
        # The pipeline fits one prior per evaluated split. When the file carries
        # the per-split priors, expose the one for the split being loaded so the
        # consistency check and the estimator report see the prior that shrunk
        # these rows. (alpha/beta at the top level hold the test-split prior.)
        if "prior_split_names" in out:
            names = [str(x) for x in out["prior_split_names"].tolist()]
            if split in names:
                i = names.index(split)
                out["alpha"] = np.asarray(out["prior_alpha"][i])
                out["beta"] = np.asarray(out["prior_beta"][i])
                out["fit_split"] = np.asarray(split)
    return out


def load_routers(path: str = "data/routers.npz", keep=None,
                 prompt_ids=None, required: bool = False):
    """Router choices, filtered to the same prompts as the rates.

    `choice` covers every prompt in the order of the unsplit rate file; `keep` is
    the boolean mask for the split being analysed. Returns None when absent unless
    required=True.
    """
    if not os.path.exists(path):
        if required:
            _require(path, "the router choice arrays")
        return None
    d = np.load(path, allow_pickle=False)
    if "choice" not in d:
        raise DataInconsistent(f"{path} must contain 'choice'; see DATA_SCHEMA.md")
    C = np.asarray(d["choice"])
    if C.ndim != 3:
        raise DataInconsistent(f"choice must be (n_routers, n_inits, n_prompts); got {C.shape}")
    out = {k: np.asarray(d[k]) for k in d.files}
    if keep is not None:
        if C.shape[-1] != len(keep):
            raise DataInconsistent(
                f"choice covers {C.shape[-1]} prompts but the rate file has {len(keep)} "
                "before splitting; these must be the same prompts in the same order"
            )
        out["choice"] = C[:, :, keep]
    if prompt_ids is not None and out["choice"].shape[-1] != len(prompt_ids):
        raise DataInconsistent(
            f"after splitting, choice has {out['choice'].shape[-1]} prompts but the rates "
            f"have {len(prompt_ids)}"
        )
    return out


def check_rates_match_verdicts(rates, verdicts, tol: float = 1e-9) -> None:
    """The check worth having: does R actually come from these verdicts?

    A shrunk rate must lie between the raw frequency and the prior mean. If any
    entry does not, the two files describe different runs, which is precisely the
    drift that produced the duplicated base rollouts the verifier audit found.
    """
    V = verdicts["verdicts"]
    R = rates["R"]
    if V.shape[:2] != R.shape:
        raise DataInconsistent(
            f"verdicts is (n={V.shape[0]}, M={V.shape[1]}, k={V.shape[2]}) but R is {R.shape}"
        )
    a, be = float(rates["alpha"]), float(rates["beta"])
    if a <= 0 or be <= 0:
        raise DataInconsistent(f"prior must be positive; got alpha={a}, beta={be}")
    prior_mean = a / (a + be)
    raw = V.mean(axis=2)
    lo = np.minimum(raw, prior_mean) - tol
    hi = np.maximum(raw, prior_mean) + tol
    bad = (R < lo) | (R > hi)
    if bad.any():
        i, j = np.argwhere(bad)[0]
        raise DataInconsistent(
            f"{int(bad.sum())} of {R.size} shrunk rates lie outside "
            f"[raw frequency, prior mean]. First at prompt {i}, member {j}: "
            f"raw={raw[i, j]:.4f}, prior={prior_mean:.4f}, R={R[i, j]:.4f}. "
            "rates.npz and verdicts.npz describe different runs."
        )


def load_all(data_dir: str = "data", split: str = "test", need_verdicts: bool = True):
    """Load everything for one split, cross-checked. The normal entry point."""
    full = load_rates(os.path.join(data_dir, "rates.npz"))          # unsplit, for the mask
    keep = full["split"] == split
    rates = load_rates(os.path.join(data_dir, "rates.npz"), split=split)
    verdicts = None
    if need_verdicts:
        verdicts = load_verdicts(os.path.join(data_dir, "verdicts.npz"), split=split)
        check_rates_match_verdicts(rates, verdicts)
    routers = load_routers(os.path.join(data_dir, "routers.npz"), keep=keep,
                           prompt_ids=rates["prompt_ids"])
    return rates, verdicts, routers


def describe(rates, verdicts=None, routers=None) -> str:
    n, M = rates["R"].shape
    a, be = float(rates["alpha"]), float(rates["beta"])
    s = [f"prompts {n}, members {M}",
         f"prior Beta({a:.3f}, {be:.3f}), strength {a + be:.3f}, mean {a / (a + be):.3f}, "
         f"fitted on {str(rates['fit_split'])}"]
    if verdicts is not None:
        s.append(f"rollouts per (prompt, member): k = {verdicts['verdicts'].shape[2]}")
        if "n_gen_tokens" in verdicts:
            t = verdicts["n_gen_tokens"].mean(axis=(0, 2))
            s.append("mean completion tokens per member: " + ", ".join(f"{x:.0f}" for x in t))
    if routers is not None:
        r, i, _ = routers["choice"].shape
        s.append(f"routers {r} families x {i} initializations")
    return "\n".join("  " + x for x in s)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "data"
    sp = sys.argv[2] if len(sys.argv) > 2 else "test"
    try:
        rates, verdicts, routers = load_all(d, sp)
    except (DataMissing, DataInconsistent) as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    print(f"loaded {d} split={sp}")
    print(describe(rates, verdicts, routers))
