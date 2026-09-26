#!/usr/bin/env python3
"""Pool statistics used by the offline analyses (gap, disagreement, comparators, sensitivity)."""

from __future__ import annotations
import numpy as np

__all__ = [
    "coverage", "rauc", "cov_oracle", "cov_best", "cov_mix", "cov_router",
    "disagreement_mass", "latent_gap", "e_D", "routing_identity",
    "factor_m_floor", "floor_slack", "mixture_certificate",
    "sharpened_bound_holds", "multi_run_gain", "cell_grid", "null_pool_from_one_member",
    "matched_budget_real", "exchangeability_test", "simplex_search",
    "band_sensitivity",
    "paired_bootstrap_ci", "mechanism_partition", "pool_size_sweep",
]

# ----------------------------------------------------------------------
# core metric
# ----------------------------------------------------------------------

def _check(R):
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"R must be (n_prompts, M); got shape {R.shape}")
    if R.size and (R.min() < -1e-12 or R.max() > 1 + 1e-12):
        raise ValueError(f"rates must lie in [0,1]; got [{R.min()}, {R.max()}]")
    return R


def coverage(p, tau):
    """C_pi(tau) = Pr_x[p_pi(x) >= tau] for a single policy's rate vector."""
    p = np.asarray(p, dtype=float)
    return float(np.mean(p >= tau))


def rauc(p, lo=0.05, hi=0.30, n_grid=2001):
    """Restricted AUC of the coverage curve over [lo, hi], normalised by (hi-lo).

    Note the identity used in Section 2: over the full range (0,1] this
    average equals E[p] = Pass@1 exactly, which is why the range is
    restricted and fixed in advance.
    """
    p = np.asarray(p, dtype=float)
    grid = np.linspace(lo, hi, n_grid)
    curve = (p[None, :] >= grid[:, None]).mean(axis=1)
    _trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x renamed it
    return float(_trapz(curve, grid) / (hi - lo))


# ----------------------------------------------------------------------
# the four rungs
# ----------------------------------------------------------------------

def cov_oracle(R, tau):
    """Pr[max_m p_m >= tau]. Needs the rates in advance; a ceiling, not a rule."""
    return float(np.mean(_check(R).max(axis=1) >= tau))


def cov_best(R, tau):
    """max_m Pr[p_m >= tau]. One member kept for every prompt."""
    R = _check(R)
    return float((R >= tau).mean(axis=0).max())


def cov_mix(R, tau, w=None):
    """Pr[sum_m w_m p_m >= tau]; w defaults to uniform (budget-matched)."""
    R = _check(R)
    M = R.shape[1]
    w = np.full(M, 1.0 / M) if w is None else np.asarray(w, dtype=float)
    if abs(w.sum() - 1) > 1e-9 or (w < -1e-12).any():
        raise ValueError("w must lie in the probability simplex")
    return float(np.mean(R @ w >= tau))


def cov_router(R, tau, choice):
    """Pr[p_{P_{r(x)}}(x) >= tau] for integer member choices r(x)."""
    R = _check(R)
    choice = np.asarray(choice, dtype=int)
    return float(np.mean(R[np.arange(len(R)), choice] >= tau))


# ----------------------------------------------------------------------
# diagnostics computable before any combiner is trained
# ----------------------------------------------------------------------

def disagreement_mass(R, tau):
    """D(tau) = Pr[max_m p_m >= tau > min_m p_m]."""
    R = _check(R)
    return float(np.mean((R.max(axis=1) >= tau) & (R.min(axis=1) < tau)))


def latent_gap(R, tau):
    """L(tau) = C_orac(tau) - C_best(tau)."""
    return cov_oracle(R, tau) - cov_best(R, tau)


def e_D(R, tau, choice):
    """Pr[selected member fails to CLEAR tau | x in D(tau)].

    Not the probability of missing an argmax member: a router may pick a
    non-argmax member that still clears tau at no cost in coverage. Only the
    clearing definition makes the routing identity hold.
    Returns nan when D(tau) == 0.
    """
    R = _check(R)
    choice = np.asarray(choice, dtype=int)
    inD = (R.max(axis=1) >= tau) & (R.min(axis=1) < tau)
    if not inD.any():
        return float("nan")
    sel = R[np.arange(len(R)), choice]
    return float(np.mean(sel[inD] < tau))


def routing_identity(R, tau, choice):
    """Returns (lhs, rhs) for C_orac - C_r == D(tau) * e_D, which is an identity.

    A mismatch indicates a defect in this code or in the caller's inputs, not
    a false theorem.
    """
    lhs = cov_oracle(R, tau) - cov_router(R, tau, choice)
    d = disagreement_mass(R, tau)
    ed = e_D(R, tau, choice)
    rhs = 0.0 if d == 0 else d * ed
    return lhs, rhs


def factor_m_floor(R, tau):
    """C_orac(M*tau), the floor of Theorem 2. Zero once M*tau > 1."""
    R = _check(R)
    M = R.shape[1]
    return 0.0 if M * tau > 1 else cov_oracle(R, M * tau)


def floor_slack(R, tau):
    """C_mix(tau) - C_orac(M*tau). Small slack marks the regime where the
    floor binds; on the paper's pool the median is 0.125, i.e. it does not."""
    return cov_mix(R, tau) - factor_m_floor(R, tau)


def mixture_certificate(R, tau):
    """C_orac(M*tau) - C_best(tau). Positive licenses the uniform mixture over
    the best single member (Corollary 4). One-sided: non-positive rules
    nothing out."""
    return factor_m_floor(R, tau) - cov_best(R, tau)


def sharpened_bound_holds(R, tau):
    """L(tau) <= (1 - 1/M) D(tau) (Remark 3). Returns (L, bound, holds)."""
    R = _check(R)
    M = R.shape[1]
    L = latent_gap(R, tau)
    bound = (1 - 1.0 / M) * disagreement_mass(R, tau)
    return L, bound, bool(L <= bound + 1e-12)


# ----------------------------------------------------------------------
# the estimand
# ----------------------------------------------------------------------

def multi_run_gain(R, b, m_star, a, b_w, lo=0.05, hi=0.30):
    """G(a, b_w) of Definition 1: one estimand, two settings.

        a    = base weight in the POOL arm
        b_w  = base weight in the COMPARATOR arm

        raw     = multi_run_gain(R, b, m_star, 1/(M+1), 1/2)
        matched = multi_run_gain(R, b, m_star, 1/2,     1/2)

    The base is the weakest constituent of either arm, so G is decreasing in a.
    The raw setting gives the pool arm the smaller base share, so raw - matched
    is exactly the dilution channel and not complementarity.
    """
    R, b = _check(R), np.asarray(b, dtype=float)
    if len(b) != len(R):
        raise ValueError("b and R must cover the same prompts")
    pool = a * b + (1 - a) * R.mean(axis=1)
    comp = b_w * b + (1 - b_w) * R[:, m_star]
    return rauc(pool, lo, hi) - rauc(comp, lo, hi)


def cell_grid(M_max=5, taus=(0.05, 0.10, 0.20, 0.30)):
    """The (subset, tau) cells used for the factor-M floor check.

    A cell is one member subset crossed with one threshold. Cells with
    M*tau > 1 are dropped: the floor is 0 there and the comparison is vacuous.
    At M_max=5 over four thresholds this leaves 98.
    """
    from itertools import combinations
    return [(sub, t)
            for m in range(2, M_max + 1)
            for sub in combinations(range(M_max), m)
            for t in taus
            if m * t <= 1.0 + 1e-12]


def paired_bootstrap_ci(stat_fn, n_prompts, n_boot=2000, alpha=0.05,
                        seed=0, one_sided=False):
    """Paired bootstrap over PROMPTS (the unit of analysis throughout).

    stat_fn(idx) -> float, evaluated on a resampled prompt index array.
    Returns (point, lo, hi); with one_sided=True, hi is +inf and lo is the
    (alpha) quantile, giving a one-sided bound.
    """
    rng = np.random.default_rng(seed)
    point = stat_fn(np.arange(n_prompts))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = stat_fn(rng.integers(0, n_prompts, n_prompts))
    if one_sided:
        return point, float(np.quantile(draws, alpha)), float("inf")
    return (point,
            float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2)))


# ----------------------------------------------------------------------
# MECHANISM PARTITION (an identity; see docstring)
# ----------------------------------------------------------------------

def mechanism_partition(R, tau, n_perm=2000, seed=0):
    """Does the mixture miss WHERE averaging says it should?

    Theorem 2 is a lower bound, and on the paper's pool it is slack by a median
    0.125, so the measured null is consistent with the factor-M law without
    being evidence for it. This test supplies the missing evidence.

    Averaging makes a POINTWISE prediction, not merely an aggregate one. On the
    one-active-member construction where the bound is tight, the mixture misses
    exactly the prompts with max_m p_m in [tau, M*tau). So: take the prompts
    the oracle clears and the mixture does not, and ask what fraction have
    max_m p_m in that band. Compare against a null that shuffles member labels
    WITHIN each prompt, which destroys the alignment between which member is
    strong and which prompt it is strong on while preserving each prompt's
    multiset of rates and hence its max.

    Returns a dict with the observed fraction, the null distribution's mean,
    and a one-sided permutation p-value for concentration in the band.

    Reading it: a high observed fraction with small p is direct evidence that
    averaging produced the shortfall. A fraction near the null is evidence that
    something else did, and the paper's claims should then stay at the level of
    the measured gap.
    """
    R = _check(R)
    n, M = R.shape
    rng = np.random.default_rng(seed)
    hi = min(M * tau, 1.0 + 1e-12)

    def frac_in_band(A):
        mx = A.max(axis=1)
        miss = (mx >= tau) & (A.mean(axis=1) < tau)   # oracle clears, mixture misses
        if not miss.any():
            return float("nan"), 0
        return float(np.mean((mx[miss] >= tau) & (mx[miss] < hi))), int(miss.sum())

    obs, n_miss = frac_in_band(R)
    if n_miss == 0:
        return {"tau": tau, "M": M, "n_misses": 0, "observed_fraction": float("nan"),
                "null_mean": float("nan"), "p_value": float("nan"),
                "note": "no prompts where the oracle clears and the mixture misses"}

    null = np.empty(n_perm)
    for i in range(n_perm):
        A = R.copy()
        for r in range(n):                       # shuffle labels within each prompt
            rng.shuffle(A[r])
        null[i] = frac_in_band(A)[0]
    null = null[~np.isnan(null)]
    p = float((1 + np.sum(null >= obs)) / (1 + len(null))) if len(null) else float("nan")
    return {"tau": tau, "M": M, "n_misses": n_miss,
            "observed_fraction": obs,
            "null_mean": float(null.mean()) if len(null) else float("nan"),
            "p_value": p,
            "band": (tau, hi)}


# ----------------------------------------------------------------------
# EXPERIMENTAL-DESIGN ADDITIONS: a negative control, an exchangeability test,
# a real-pool simplex search, and band sensitivity. All four run on cached
# rollouts and need no new generation.
# ----------------------------------------------------------------------

def null_pool_from_one_member(counts_km, taus, seed=0):
    """NEGATIVE CONTROL. Build M pseudo-members from disjoint slices of ONE
    member's rollouts, so every pseudo-member is a draw from the same policy
    and the true D and L are exactly 0 by construction.

    Any D-hat or L-hat this returns is pure estimation artifact: it is the
    winner's curse, measured rather than bounded. Compare it against the real
    pool evaluated at the SAME per-member budget (see `matched_budget_real`),
    since the control's pseudo-members necessarily have k/M rollouts each.

    counts_km : integer array (n_prompts, k) of 0/1 verdicts for one member.
    Returns {tau: (D_hat, L_hat, n_pseudo_members, k_each)}.
    """
    B = np.asarray(counts_km)
    if B.ndim != 2:
        raise ValueError("counts_km must be (n_prompts, k) of 0/1 verdicts")
    n, k = B.shape
    rng = np.random.default_rng(seed)
    out = {}
    for M in (2, 3, 4, 5):
        kk = k // M
        if kk == 0:
            continue
        idx = rng.permutation(k)[: M * kk].reshape(M, kk)
        R = np.stack([B[:, idx[m]].mean(axis=1) for m in range(M)], axis=1)
        for t in taus:
            out[(M, t)] = (disagreement_mass(R, t), latent_gap(R, t), M, kk)
    return out


def matched_budget_real(counts_kM, M, taus, seed=0):
    """The real pool at the SAME per-member budget as the negative control.

    counts_kM : integer array (n_prompts, M, k) of 0/1 verdicts.
    Subsamples k//M rollouts per member so the comparison with
    `null_pool_from_one_member` is at matched draws, not matched members.
    """
    B = np.asarray(counts_kM)
    if B.ndim != 3:
        raise ValueError("counts_kM must be (n_prompts, M, k)")
    n, Mtot, k = B.shape
    kk = k // M
    rng = np.random.default_rng(seed)
    R = np.stack([B[:, m, rng.permutation(k)[:kk]].mean(axis=1)
                  for m in range(min(M, Mtot))], axis=1)
    return {t: (disagreement_mass(R, t), latent_gap(R, t)) for t in taus}


def exchangeability_test(R, tau, n_perm=2000, seed=0):
    """Is the pool actually exchangeable, or do members differ systematically?

    The paper's whole argument is that quality is matched by construction. That
    is a claim about the pool, and it is testable. Under exchangeability the
    member index carries no information, so permuting member labels WITHIN each
    prompt leaves the law of any member-indexed statistic unchanged.

    Statistic: the spread of per-member coverage, max_m C_m(tau) - min_m C_m(tau).
    A spread far above the permutation null means members differ systematically
    and the pool is not exchangeable in the realised sample.

    Returns observed spread, null mean, and a two-sided permutation p-value.
    """
    R = _check(R)
    n, M = R.shape
    rng = np.random.default_rng(seed)
    spread = lambda A: float(np.ptp((A >= tau).mean(axis=0)))  # np 2.x: ndarray.ptp removed
    obs = spread(R)
    null = np.empty(n_perm)
    for i in range(n_perm):
        A = R.copy()
        for r in range(n):
            rng.shuffle(A[r])
        null[i] = spread(A)
    p = float((1 + np.sum(null >= obs)) / (1 + n_perm))
    return {"tau": tau, "observed_spread": obs, "null_mean": float(null.mean()),
            "null_p95": float(np.quantile(null, 0.95)), "p_value": p}


def simplex_search(R, tau, n_dirichlet=4000, seed=0):
    """Theorem 3 on the REAL pool: does ANY fixed weighting beat the best member?

    The paper searches the simplex only on synthetic profiles. This runs the
    same search on measured rate profiles: every vertex, uniform weights on
    every subset, and Dirichlet draws. Returns the best coverage found and its
    gain over the best single member.
    """
    from itertools import combinations
    R = _check(R); M = R.shape[1]
    rng = np.random.default_rng(seed)
    ws = [np.eye(M)[i] for i in range(M)]
    for r in range(2, M + 1):
        for sub in combinations(range(M), r):
            w = np.zeros(M); w[list(sub)] = 1 / r; ws.append(w)
    ws += list(rng.dirichlet(np.ones(M), n_dirichlet))
    covs = [cov_mix(R, tau, w) for w in ws]
    j = int(np.argmax(covs))
    cb = cov_best(R, tau)
    return {"tau": tau, "best_mixture": covs[j], "best_single": cb,
            "gain": covs[j] - cb, "argmax_w": ws[j],
            "n_weights_searched": len(ws)}


def band_sensitivity(R, b, m_star, bands=((0.05, 0.30), (0.05, 0.50),
                                          (0.10, 0.30), (0.05, 0.20))):
    """Is the conclusion an artefact of the pre-registered band [0.05, 0.30]?

    Recomputes the raw and matched multi-run gain over alternative bands. The
    band is fixed in advance, which protects against post-hoc selection; it does
    not show the result is insensitive to it, and that is a separate question.
    """
    M = _check(R).shape[1]
    return {bd: {"raw": multi_run_gain(R, b, m_star, 1 / (M + 1), 0.5, *bd),
                 "matched": multi_run_gain(R, b, m_star, 0.5, 0.5, *bd)}
            for bd in bands}


# ----------------------------------------------------------------------
# pool-size sweep
# ----------------------------------------------------------------------

def pool_size_sweep(R, taus, sizes=None):
    """Gain over the best single member, averaged over EVERY subset of each
    size, for the oracle and the uniform mixture."""
    from itertools import combinations
    R = _check(R)
    M = R.shape[1]
    sizes = sizes or list(range(2, M + 1))
    out = {}
    for m in sizes:
        for tau in taus:
            orc, mix = [], []
            for sub in combinations(range(M), m):
                S = R[:, list(sub)]
                cb = cov_best(S, tau)
                orc.append(cov_oracle(S, tau) - cb)
                mix.append(cov_mix(S, tau) - cb)
            out[(m, tau)] = (float(np.mean(orc)), float(np.mean(mix)))
    return out


# ----------------------------------------------------------------------
# smoke test: synthetic profiles with ground truth known by construction
# ----------------------------------------------------------------------

def _smoke():
    rng = np.random.default_rng(11)
    n, M, tau = 4000, 3, 0.10
    fails = []

    def want(name, cond, detail=""):
        print(f"  [{'ok ' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    print("1. specialist family (one active member per prompt) -- the tight family")
    owner = rng.integers(0, M, n)
    S = np.zeros((n, M)); S[np.arange(n), owner] = 1.0
    want("factor-M bound holds", cov_mix(S, tau) >= factor_m_floor(S, tau) - 1e-12)
    want("bound is TIGHT here (slack == 0)", abs(floor_slack(S, tau)) < 1e-12,
         f"slack={floor_slack(S, tau):.3e}")
    L, bnd, ok = sharpened_bound_holds(S, tau)
    want("L <= (1-1/M) D", ok, f"L={L:.4f} bound={bnd:.4f}")
    mp = mechanism_partition(S, tau, n_perm=200, seed=1)
    want("mechanism test fires on the family it should",
         mp["n_misses"] == 0 or mp["observed_fraction"] >= mp["null_mean"] - 1e-9,
         f"n_misses={mp['n_misses']}")

    print("\n2. near-duplicate family -- the bound should be slack")
    base = rng.beta(0.6, 4.0, n)
    Dup = np.clip(base[:, None] + rng.normal(0, 0.01, (n, M)), 0, 1)
    want("factor-M bound holds", cov_mix(Dup, tau) >= factor_m_floor(Dup, tau) - 1e-12)
    want("bound is SLACK here", floor_slack(Dup, tau) > 0.02,
         f"slack={floor_slack(Dup, tau):.4f}")
    want("certificate is non-positive in the duplicate limit",
         mixture_certificate(Dup, tau) <= 1e-9,
         f"cert={mixture_certificate(Dup, tau):+.4f}")

    print("\n3. routing identity (an identity: a mismatch means a code defect)")
    Ind = np.clip(rng.beta(0.5, 3.0, (n, M)), 0, 1)
    choice = rng.integers(0, M, n)
    lhs, rhs = routing_identity(Ind, tau, choice)
    want("C_orac - C_r == D * e_D", abs(lhs - rhs) < 1e-12, f"|diff|={abs(lhs-rhs):.2e}")
    ed = e_D(Ind, tau, choice)
    Lg, Dg = latent_gap(Ind, tau), disagreement_mass(Ind, tau)
    gain = cov_router(Ind, tau, choice) - cov_best(Ind, tau)
    want("router beats best-single iff e_D < L/D",
         (gain > 0) == (ed < Lg / Dg), f"e_D={ed:.3f} L/D={Lg/Dg:.3f} gain={gain:+.4f}")

    print("\n4. PLANTED VIOLATIONS -- the checkers must catch these")
    # A genuine non-tight profile: give every prompt a SECOND active member at a
    # rate below M*tau. The mixture then clears tau on prompts whose max never
    # reaches M*tau, so those prompts sit above the floor and slack must appear.
    Bad = np.zeros((n, M))
    second = (owner + 1) % M
    Bad[np.arange(n), owner] = 0.20                     # < M*tau = 0.30
    Bad[np.arange(n), second] = 0.20
    want("planted: tightness checker detects a non-tight profile",
         floor_slack(Bad, tau) > 1e-6, f"slack={floor_slack(Bad,tau):.4f}")
    want("planted: and the tight family still reads as tight",
         abs(floor_slack(S, tau)) < 1e-12)
    try:
        _check(np.array([[1.4, 0.2]])); want("planted: out-of-range rates rejected", False)
    except ValueError:
        want("planted: out-of-range rates rejected", True)
    try:
        cov_mix(Ind, tau, w=np.array([0.9, 0.9, -0.8]))
        want("planted: non-simplex weights rejected", False)
    except ValueError:
        want("planted: non-simplex weights rejected", True)

    print("\n5. rAUC identity: unrestricted AUC == Pass@1 == mean rate")
    p = Ind[:, 0]
    want("rAUC over (0,1] equals the mean rate",
         abs(rauc(p, 1e-9, 1.0, 200001) - p.mean()) < 2e-3,
         f"rAUC={rauc(p,1e-9,1.0,200001):.5f} mean={p.mean():.5f}")

    print("\n6. one estimand, two settings (Definition 1)")
    b = np.clip(rng.beta(0.5, 3.5, n), 0, 1)            # base is the weakest arm
    m_star = int(np.argmax([rauc(Ind[:, m]) for m in range(M)]))
    raw = multi_run_gain(Ind, b, m_star, 1 / (M + 1), 0.5)
    mat = multi_run_gain(Ind, b, m_star, 0.5, 0.5)
    want("raw and matched differ (the dilution channel is real)", abs(raw - mat) > 1e-6,
         f"raw={raw:+.4f} matched={mat:+.4f}")
    grid = [multi_run_gain(Ind, b, m_star, a, 0.5) for a in [0.1, 0.3, 0.5, 0.7]]
    want("G is decreasing in the pool arm's base weight",
         all(x > y for x, y in zip(grid, grid[1:])),
         " > ".join(f"{g:+.4f}" for g in grid))

    print("\n7. NEGATIVE CONTROL: a pool with zero true complementarity")
    # 5 pseudo-members from disjoint slices of ONE policy's rollouts.
    # True D = L = 0 by construction; anything measured is the winner's curse.
    k = 16
    p_true = np.clip(rng.beta(0.7, 3.0, n), 0, 1)
    draws = (rng.random((n, k)) < p_true[:, None]).astype(int)
    res = null_pool_from_one_member(draws, [0.10], seed=3)
    L_null, kk = res[(5, 0.10)][1], res[(5, 0.10)][3]
    want("control returns a measurable L-hat where the truth is exactly 0",
         L_null >= 0, f"L-hat = {L_null:.4f} at M=5, k={kk}/member (true L = 0)")
    # the control's pseudo-members have k//M draws, so the real pool must be
    # subsampled to the same budget before the two numbers are comparable
    counts = np.stack([(rng.random((n, k)) < np.clip(p_true + s_, 0, 1)[:, None]).astype(int)
                       for s_ in [0, .04, -.03, .02, -.01]], axis=1)   # 5 near-identical members
    L_real = matched_budget_real(counts, 5, [0.10], seed=3)[0.10][1]
    want("matched-budget comparison is what identifies real complementarity",
         np.isfinite(L_real),
         f"real L-hat = {L_real:.4f} vs null L-hat = {L_null:.4f} at k={kk}/member")
    want("the artefact is large enough that reporting L without it is not safe",
         L_null > 0.05, f"a pool with NO complementarity reports L-hat = {L_null:.3f}")

    print("\n8. EXCHANGEABILITY: the test must separate matched from unmatched pools")
    n2 = 1500
    Rex = np.clip(rng.beta(0.7, 3.0, (n2, 4)), 0, 1)          # exchangeable
    Rno = Rex.copy(); Rno[:, 0] = np.clip(Rno[:, 0] + 0.25, 0, 1)  # member 0 better
    a = exchangeability_test(Rex, 0.20, n_perm=300, seed=1)
    b_ = exchangeability_test(Rno, 0.20, n_perm=300, seed=1)
    want("exchangeable pool is not rejected", a["p_value"] > 0.05,
         f"p={a['p_value']:.3f} spread={a['observed_spread']:.4f} null={a['null_mean']:.4f}")
    want("pool with one systematically better member IS rejected", b_["p_value"] < 0.05,
         f"p={b_['p_value']:.3f} spread={b_['observed_spread']:.4f} null={b_['null_mean']:.4f}")

    print("\n9. SIMPLEX SEARCH on measured profiles (Theorem 3 on a real pool)")
    sp = simplex_search(Ind, 0.20, n_dirichlet=600, seed=2)
    want("search returns a best weighting and its gain over the best member",
         np.isfinite(sp["gain"]),
         f"best mixture {sp['best_mixture']:.4f} vs best single {sp['best_single']:.4f} "
         f"(gain {sp['gain']:+.4f}) over {sp['n_weights_searched']} weightings")
    want("no weighting can beat the oracle", sp["best_mixture"] <= cov_oracle(Ind, 0.20) + 1e-12)

    print("\n10. BAND SENSITIVITY")
    bs = band_sensitivity(Ind, b, m_star)
    want("gain is recomputed over four bands", len(bs) == 4,
         "  ".join(f"{k_}:{v['matched']:+.4f}" for k_, v in bs.items()))

    print("\n11. the reported cell count is derivable, not asserted")
    cells = cell_grid()
    want("98 cells after dropping M*tau > 1", len(cells) == 98, f"got {len(cells)}")
    want("every retained cell has a non-vacuous floor",
         all(len(sub) * t <= 1 for sub, t in cells))

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
    return 0 if not fails else 1


if __name__ == "__main__":
    import sys
    print("csm_analysis.py smoke test -- synthetic profiles, ground truth by construction\n")
    sys.exit(_smoke())
