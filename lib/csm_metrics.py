"""Coverage, rAUC, oracle, uniform mixture and router coverage from per-prompt success rates.
Run directly for its self-tests."""

from __future__ import annotations

import numpy as np

__all__ = [
    "cover_at_tau", "cover_curve", "restricted_auc",
    "disagreement_rate", "latent_gap", "mixture_floor", "mixture_certificate",
    "breadth_height_bound", "routing_loss", "teacher_mixture_rates",
    "mixture_rates", "oracle_rates", "router_rates", "pass_at_k",
    "best_single_index", "g_multi", "recovery_fractions",
    "clopper_pearson", "eb_beta_binomial_prior", "posterior_rates",
    "bootstrap_ci", "simultaneous_band", "paired_bootstrap_gap",
    "non_inferiority", "adaptive_allocation_targets",
    "synth_success_counts",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 1. Coverage metrics.  Cover@tau is a functional of the RATE vector only.
# ---------------------------------------------------------------------------

def cover_at_tau(rates: np.ndarray, tau: float, weights: np.ndarray | None = None) -> float:
    """Cover@tau = Pr_x[ p(x) >= tau ]  (Dragoi et al., 2025).

    `rates` is a length-n_items vector of per-prompt success rates.
    No sampling budget appears anywhere in this function: that is Prop. 1.
    """
    rates = np.asarray(rates, dtype=float)
    if rates.ndim != 1:
        raise ValueError("cover_at_tau expects a 1-D rate vector")
    hit = (rates >= tau - _EPS).astype(float)
    if weights is None:
        return float(hit.mean())
    w = np.asarray(weights, dtype=float)
    if w.shape != rates.shape:
        raise ValueError("weights must match rates")
    return float(np.sum(w * hit) / np.sum(w))


def cover_curve(rates: np.ndarray, taus: np.ndarray,
                weights: np.ndarray | None = None) -> np.ndarray:
    """Cover@tau evaluated on a grid of tau."""
    return np.array([cover_at_tau(rates, t, weights) for t in np.asarray(taus, float)])


def restricted_auc(rates: np.ndarray, tau_lo: float, tau_hi: float,
                   n_grid: int = 513, weights: np.ndarray | None = None) -> float:
    """Normalised restricted area under the Cover@tau curve on [tau_lo, tau_hi].

    Normalised so that a model clearing tau everywhere on the range scores 1.0.
    The tau range must be declared before looking at test data.
    """
    if not (0.0 < tau_lo < tau_hi <= 1.0):
        raise ValueError("require 0 < tau_lo < tau_hi <= 1")
    taus = np.linspace(tau_lo, tau_hi, int(n_grid))
    curve = cover_curve(rates, taus, weights)
    return float(np.trapezoid(curve, taus) / (tau_hi - tau_lo))


def pass_at_k(rates: np.ndarray, k: int) -> float:
    """Pass@k = E_x[ 1 - (1-p(x))^k ].  Unlike Cover@tau this DOES depend on k."""
    rates = np.asarray(rates, float)
    return float(np.mean(1.0 - np.power(1.0 - rates, int(k))))


# ---------------------------------------------------------------------------
# 2. Pool combination rules: the four rungs.
# ---------------------------------------------------------------------------

def mixture_rates(rate_matrix: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Rung 2. Per-prompt rate of the token-budget-matched mixture sum_m w_m p_m.

    A uniform ensemble that splits one total budget across members has exactly this
    per-prompt rate at EVERY total budget (Prop. 1).
    """
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    if weights is None:
        w = np.full(R.shape[0], 1.0 / R.shape[0])
    else:
        w = np.asarray(weights, float)
        if w.shape[0] != R.shape[0]:
            raise ValueError("weights must have one entry per model")
        if np.any(w < -_EPS) or abs(w.sum() - 1.0) > 1e-8:
            raise ValueError("weights must lie in the probability simplex")
    return w @ R


def oracle_rates(rate_matrix: np.ndarray) -> np.ndarray:
    """Rung 1 (latent). Per-prompt maximum over parents. Uses labels: diagnostic only."""
    return np.max(np.atleast_2d(np.asarray(rate_matrix, float)), axis=0)


def router_rates(rate_matrix: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    """Rung 3. Per-prompt rate of the parent chosen by a router.

    `assignment[i]` is the model index selected for item i.  A router trained
    without test labels will generally not match `argmax`, which is the point.
    """
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    a = np.asarray(assignment, int)
    if a.shape[0] != R.shape[1]:
        raise ValueError("assignment must have one entry per item")
    if a.min() < 0 or a.max() >= R.shape[0]:
        raise ValueError("assignment indices out of range")
    return R[a, np.arange(R.shape[1])]


# ---------------------------------------------------------------------------
# 3b. The exploitability calculus (Theorems 1-4).
# ---------------------------------------------------------------------------

def disagreement_rate(rate_matrix: np.ndarray, tau: float) -> float:
    """D(tau) = Pr[max_m p_m >= tau > min_m p_m]  (Theorem 1).

    The probability that the choice of member matters at reliability tau. One
    pass over cached rollouts; no router, no training, no labels beyond the
    verifier. Upper-bounds the latent gap and hence every rung's gain.
    """
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    hi = np.max(R, axis=0) >= tau - _EPS
    lo = np.min(R, axis=0) < tau - _EPS
    return float(np.mean(hi & lo))


def latent_gap(rate_matrix: np.ndarray, tau: float) -> float:
    """L(tau) = C_orac(tau) - max_m C_m(tau)."""
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    return cover_at_tau(oracle_rates(R), tau) - max(
        cover_at_tau(R[m], tau) for m in range(R.shape[0]))


def mixture_floor(rate_matrix: np.ndarray, tau: float) -> float:
    """C_orac(M*tau): the guaranteed lower bound on the uniform mixture (Thm 2).

    Returns nan when M*tau > 1, where the bound is vacuous.
    """
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    M = R.shape[0]
    if M * tau > 1.0:
        return float("nan")
    return cover_at_tau(oracle_rates(R), M * tau)


def mixture_certificate(rate_matrix: np.ndarray, tau: float) -> dict:
    """Corollary 1. A positive certificate GUARANTEES uniform mixing helps.

    certificate = C_orac(M*tau) - C_best(tau).  Sufficient, not necessary.
    Computable before the ensemble is built.
    """
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    best = max(cover_at_tau(R[m], tau) for m in range(R.shape[0]))
    floor = mixture_floor(R, tau)
    cert = floor - best if np.isfinite(floor) else float("nan")
    return {"certificate": cert,
            "guaranteed": bool(np.isfinite(cert) and cert > 0.0),
            "floor": floor, "best_single": best,
            "realised_gain": cover_at_tau(mixture_rates(R), tau) - best}


def breadth_height_bound(rate_matrix: np.ndarray, tau: float,
                         s: float = 0.0) -> float:
    """Upper bound of Theorem 2: Pr[N_s >= M(tau-s)/(max_m p_m - s)]."""
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    M = R.shape[0]
    if not (0.0 <= s < tau):
        raise ValueError("require 0 <= s < tau")
    h = np.max(R, axis=0)
    Ns = (R >= s - _EPS).sum(axis=0)
    need = M * (tau - s) / np.maximum(h - s, _EPS)
    return float(np.mean(Ns >= need - 1e-9))


def routing_loss(rate_matrix: np.ndarray, assignment: np.ndarray,
                 tau: float) -> dict:
    """Theorem 3. Coverage loss vs the oracle, decomposed on D(tau).

    Returns the exact loss, the disagreement rate D, the router's error rate
    restricted to D, their product (which equals the loss), and the break-even
    accuracy 1 - L/D above which the router beats the best single member.
    """
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    rr = router_rates(R, assignment)
    hi = np.max(R, axis=0) >= tau - _EPS
    miss = rr < tau - _EPS
    loss = float(np.mean(hi & miss))
    D = disagreement_rate(R, tau)
    inD = hi & (np.min(R, axis=0) < tau - _EPS)
    e_D = float(np.mean(miss[inD])) if inD.any() else 0.0
    L = latent_gap(R, tau)
    return {"loss": loss, "D": D, "e_D": e_D, "D_times_eD": D * e_D,
            "identity_holds": abs(loss - D * e_D) < 1e-9,
            "break_even_accuracy": (1.0 - L / D) if D > _EPS else float("nan"),
            "beats_best_single": bool(D > _EPS and e_D < L / D - 1e-12)}


def teacher_mixture_rates(rate_matrix: np.ndarray,
                          lam: np.ndarray) -> np.ndarray:
    """Theorem 4. p_S(x) = sum_m lambda_m(x) p_m(x) in the teacher-mixture limit.

    `lam` has shape (n_models, n_items) with columns on the simplex.
    """
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    L = np.atleast_2d(np.asarray(lam, float))
    if L.shape != R.shape:
        raise ValueError("lam must match rate_matrix")
    if np.any(L < -_EPS) or np.max(np.abs(L.sum(axis=0) - 1.0)) > 1e-8:
        raise ValueError("lam columns must lie on the simplex")
    return np.einsum('mn,mn->n', L, R)


# ---------------------------------------------------------------------------
# 3. The estimand.
# ---------------------------------------------------------------------------

def best_single_index(rate_matrix: np.ndarray, base_rates: np.ndarray | None,
                      tau_lo: float, tau_hi: float) -> int:
    """Index m* maximising rAUC of base (+) single parent.

    Called on TEST rates for the conservative comparator of Definition 1, which
    gives the comparator an optimistic advantage on purpose.
    """
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    scores = []
    for m in range(R.shape[0]):
        if base_rates is None:
            r = R[m]
        else:
            r = mixture_rates(np.vstack([np.asarray(base_rates, float), R[m]]))
        scores.append(restricted_auc(r, tau_lo, tau_hi))
    return int(np.argmax(scores))


def g_multi(rate_matrix: np.ndarray, base_rates: np.ndarray | None,
            tau_lo: float, tau_hi: float,
            comparator_index: int | None = None) -> dict:
    """Conservative multi-seed gain (Definition 1).

    Returns the pool rAUC, the comparator rAUC, the gap, and the comparator used.
    If `comparator_index` is None the TEST-best single parent is used (hostile to
    our own claim).  Pass a validation-selected index for the secondary variant.
    """
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    members = R if base_rates is None else np.vstack([np.asarray(base_rates, float), R])
    pool = restricted_auc(mixture_rates(members), tau_lo, tau_hi)

    m = best_single_index(R, base_rates, tau_lo, tau_hi) if comparator_index is None \
        else int(comparator_index)
    comp_members = R[m][None, :] if base_rates is None else \
        np.vstack([np.asarray(base_rates, float), R[m]])
    comp = restricted_auc(mixture_rates(comp_members), tau_lo, tau_hi)
    return {"pool_rauc": pool, "comparator_rauc": comp,
            "g_multi": pool - comp, "comparator_index": m,
            "comparator_selected_on": "test" if comparator_index is None else "validation"}


def recovery_fractions(latent_gap: float, witness_gap: float,
                       student_gap: float,
                       latent_gap_ci: tuple[float, float] | None = None) -> dict:
    """rho_W = W / L and rho_S = S / L.

    Suppressed (returned as None) when the latent gap's interval includes zero,
    because a ratio with an indistinguishable-from-zero denominator is
    uninterpretable.  This guard is intentional, not defensive padding.
    """
    suppress = latent_gap <= _EPS or (
        latent_gap_ci is not None and latent_gap_ci[0] <= 0.0 <= latent_gap_ci[1])
    if suppress:
        return {"latent_gap": latent_gap, "rho_W": None, "rho_S": None,
                "suppressed": True,
                "reason": "latent gap not distinguishable from zero"}
    return {"latent_gap": latent_gap,
            "rho_W": witness_gap / latent_gap,
            "rho_S": student_gap / latent_gap,
            "suppressed": False, "reason": None}


# ---------------------------------------------------------------------------
# 4. Interval estimation under a finite rollout budget.
# ---------------------------------------------------------------------------

def _betainc_reg(a: float, b: float, x: float, n: int = 200_001) -> float:
    """Regularised incomplete beta I_x(a,b) by quadrature. numpy only.

    Accurate to ~1e-8 for the (a, b) magnitudes used here.  Substituting
    scipy.special.betainc gives the same answer; the smoke test checks this when
    scipy happens to be present.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    t = np.linspace(0.0, 1.0, n)
    # integrate on [0,1] then rescale, using log-space for the kernel
    def kernel(u, hi):
        u = np.clip(u * hi, 1e-300, 1.0 - 1e-16)
        return np.exp((a - 1.0) * np.log(u) + (b - 1.0) * np.log1p(-u))
    num = np.trapezoid(kernel(t, x), t) * x
    den = np.trapezoid(kernel(t, 1.0), t)
    return float(np.clip(num / den, 0.0, 1.0))


def clopper_pearson(successes: int, draws: int, alpha: float = 0.05,
                    side: str = "two") -> tuple[float, float]:
    """Exact binomial confidence bounds (Clopper & Pearson, 1934).

    side='upper' returns (0, U) with one-sided coverage 1-alpha.  The zero- and
    full-success cases use the closed forms. For example,
    0 successes in 32 draws gives U = 1 - alpha**(1/32) ~= 0.0894.
    """
    s, n = int(successes), int(draws)
    if n <= 0 or not (0 <= s <= n):
        raise ValueError("require 0 <= successes <= draws and draws > 0")
    a = alpha if side in ("upper", "lower") else alpha / 2.0

    if s == 0:
        lo = 0.0
    else:
        lo = _invert_beta(s, n - s + 1, a)
    if s == n:
        hi = 1.0
    else:
        hi = _invert_beta(s + 1, n - s, 1.0 - a)

    # closed forms, exact, no quadrature needed
    if s == 0:
        hi = 1.0 - a ** (1.0 / n)
    if s == n:
        lo = a ** (1.0 / n)

    if side == "upper":
        return 0.0, float(hi)
    if side == "lower":
        return float(lo), 1.0
    return float(lo), float(hi)


def _invert_beta(a: float, b: float, q: float, iters: int = 60) -> float:
    """Bisect the regularised incomplete beta to find x with I_x(a,b) = q."""
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _betainc_reg(a, b, mid, n=20_001) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def eb_beta_binomial_prior(succ: np.ndarray, draws: np.ndarray) -> tuple[float, float]:
    """Empirical-Bayes Beta(a, b) prior fitted across items by moment matching.

    The beta-binomial is the predictive distribution for future counts, so a prior
    fitted across items is what licenses shrunk per-item posteriors.
    Returns (a, b).  Falls back to a weak Jeffreys-like prior when the observed
    dispersion is at or below the binomial floor, in which case shrinkage is not
    identified from the data and we say so rather than inventing a prior.
    """
    s = np.asarray(succ, float).ravel()
    d = np.asarray(draws, float).ravel()
    if np.any(d <= 0):
        raise ValueError("all items need at least one draw")
    p = s / d
    m = float(np.mean(p))
    v = float(np.var(p, ddof=1)) if p.size > 1 else 0.0
    # expected within-item binomial variance
    v_bin = float(np.mean(p * (1.0 - p) / d))
    v_between = v - v_bin
    if not np.isfinite(v_between) or v_between <= 1e-9 or m <= 0.0 or m >= 1.0:
        return 0.5, 0.5  # not identified from these data
    nu = m * (1.0 - m) / v_between - 1.0
    nu = max(nu, 1e-6)
    return float(m * nu), float((1.0 - m) * nu)


def posterior_rates(succ: np.ndarray, draws: np.ndarray,
                    prior: tuple[float, float] | None = None,
                    mode: str = "mean") -> np.ndarray:
    """Shrunk per-item, per-model rate estimates from a Beta-binomial posterior.

    `mode='mean'` returns the posterior mean (a+s)/(a+b+n).  This is the estimator
    used for point reporting; interval reporting uses `posterior_interval`.
    """
    s = np.asarray(succ, float)
    d = np.asarray(draws, float)
    if s.shape != d.shape:
        raise ValueError("succ and draws must have the same shape")
    a, b = eb_beta_binomial_prior(s, d) if prior is None else prior
    if mode == "mean":
        return (a + s) / (a + b + d)
    if mode == "mode":
        num = np.maximum(a + s - 1.0, 0.0)
        den = np.maximum(a + b + d - 2.0, _EPS)
        return num / den
    raise ValueError("mode must be 'mean' or 'mode'")


def posterior_interval(succ: int, draws: int, prior: tuple[float, float],
                       alpha: float = 0.05) -> tuple[float, float]:
    """Equal-tailed Beta posterior interval for one item-model pair."""
    a, b = prior
    return (_invert_beta(a + succ, b + draws - succ, alpha / 2.0),
            _invert_beta(a + succ, b + draws - succ, 1.0 - alpha / 2.0))


def adaptive_allocation_targets(succ: np.ndarray, draws: np.ndarray,
                                taus: np.ndarray,
                                prior: tuple[float, float] | None = None,
                                alpha: float = 0.05) -> np.ndarray:
    """Items whose posterior interval straddles a declared tau, for extra draws.

    Returns a boolean mask over items (True = allocate more rollouts).  It spends
    budget only where it can change a conclusion.
    """
    s = np.asarray(succ, float)
    d = np.asarray(draws, float)
    a, b = eb_beta_binomial_prior(s, d) if prior is None else prior
    n_models, n_items = s.shape
    mask = np.zeros(n_items, dtype=bool)
    for i in range(n_items):
        for m in range(n_models):
            lo, hi = posterior_interval(int(s[m, i]), int(d[m, i]), (a, b), alpha)
            if np.any((np.asarray(taus) > lo) & (np.asarray(taus) < hi)):
                mask[i] = True
                break
    return mask


# ---------------------------------------------------------------------------
# 5. Uncertainty over prompts.
# ---------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap over ITEMS (the unit of analysis). Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    V = np.asarray(values)
    n = V.shape[-1]
    point = float(stat_fn(V))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = stat_fn(V[..., idx])
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def simultaneous_band(rates: np.ndarray, taus: np.ndarray, n_boot: int = 2000,
                      alpha: float = 0.05, seed: int = 0) -> dict:
    """Sup-t simultaneous band for the whole Cover@tau curve.

    Pointwise intervals understate the risk of reading a curve at a tau chosen
    after the fact, so the paper reports this instead.
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(rates, float)
    n = r.shape[0]
    taus = np.asarray(taus, float)
    point = cover_curve(r, taus)
    boot = np.empty((n_boot, taus.size))
    for b in range(n_boot):
        boot[b] = cover_curve(r[rng.integers(0, n, n)], taus)
    se = boot.std(axis=0, ddof=1)
    # A curve with no bootstrap variability at any tau (e.g. an exactly constant
    # rate vector) has a genuinely degenerate band.  Report that rather than
    # returning a zero-width band that looks like an implausibly precise result.
    live = se > 1e-12
    degenerate = not bool(live.any())
    if degenerate:
        return {"taus": taus, "point": point, "lo": point.copy(), "hi": point.copy(),
                "crit": float("nan"), "degenerate": True,
                "reason": "statistic has no bootstrap variability across items"}
    se_safe = np.where(live, se, np.inf)          # excluded taus contribute 0 to sup
    sup_t = np.max(np.abs(boot - point) / se_safe, axis=1)
    crit = float(np.quantile(sup_t, 1.0 - alpha))
    return {"taus": taus, "point": point,
            "lo": np.clip(point - crit * se, 0.0, 1.0),
            "hi": np.clip(point + crit * se, 0.0, 1.0),
            "crit": crit, "degenerate": False, "reason": None}


def paired_bootstrap_gap(pool_rates: np.ndarray, comp_rates: np.ndarray,
                         tau_lo: float, tau_hi: float, n_boot: int = 2000,
                         alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired bootstrap over items for the rAUC gap, with a one-sided lower bound.

    The gate in the paper is on the LOWER confidence bound, not the point estimate.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(pool_rates, float)
    b = np.asarray(comp_rates, float)
    if a.shape != b.shape:
        raise ValueError("paired inputs must align item-by-item")
    n = a.shape[0]
    point = restricted_auc(a, tau_lo, tau_hi) - restricted_auc(b, tau_lo, tau_hi)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = (restricted_auc(a[idx], tau_lo, tau_hi)
                    - restricted_auc(b[idx], tau_lo, tau_hi))
    lo2, hi2 = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"gap": float(point), "ci_lo": float(lo2), "ci_hi": float(hi2),
            "one_sided_lcb": float(np.quantile(draws, alpha)),
            "positive_lcb": bool(np.quantile(draws, alpha) > 0.0)}


def non_inferiority(student_rates: np.ndarray, anchor_rates: np.ndarray,
                    margin: float, n_boot: int = 2000, alpha: float = 0.05,
                    seed: int = 0) -> dict:
    """Interval-based Pass@1 non-inferiority: is student >= anchor - margin?

    Declares non-inferiority only if the lower bound on the mean difference lies
    above -margin.  A coverage gain purchased with an accuracy loss must not pass.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    rng = np.random.default_rng(seed)
    d = np.asarray(student_rates, float) - np.asarray(anchor_rates, float)
    n = d.shape[0]
    draws = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lcb = float(np.quantile(draws, alpha))
    return {"mean_diff": float(d.mean()), "one_sided_lcb": lcb,
            "margin": float(margin), "non_inferior": bool(lcb > -margin)}


# ---------------------------------------------------------------------------
# 6. Synthetic data. Explicitly named, never a silent fallback.
# ---------------------------------------------------------------------------

def synth_success_counts(rate_matrix: np.ndarray, k: int, seed: int = 0):
    """Draw binomial counts from KNOWN rates. For testing the estimators only.

    Any artifact produced from this function must be written with a 'SYNTH_'
    filename prefix by the caller so it can never be mistaken for a real run.
    """
    rng = np.random.default_rng(seed)
    R = np.atleast_2d(np.asarray(rate_matrix, float))
    draws = np.full(R.shape, int(k))
    succ = rng.binomial(draws, R)
    return succ, draws


# ---------------------------------------------------------------------------
# 7. Smoke test: verifies the paper's algebra numerically.
# ---------------------------------------------------------------------------

def _smoke() -> None:
    ok, fail = 0, 0

    def check(name, cond, detail=""):
        nonlocal ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {name}")
        else:
            fail += 1
            print(f"  FAIL  {name}  {detail}")

    print("\n[1] Theorems 1-4 over randomised profiles (4 families)")

    def draw(family, M, n, rg):
        if family == "independent-lowrate":
            return rg.beta(0.5, 4.0, (M, n))
        if family == "specialist":
            R = np.full((M, n), 1e-3)
            R[rg.integers(0, M, n), np.arange(n)] = rg.uniform(0.3, 1.0, n)
            return R
        if family == "near-duplicate":
            b = rg.beta(1.0, 3.0, n)
            return np.clip(b[None, :] + rg.normal(0, 0.03, (M, n)), 1e-6, 1.0)
        b = rg.beta(1.0, 6.0, n)
        sp = (rg.random((M, n)) < 0.15) * rg.uniform(0.3, 0.9, (M, n))
        return np.clip(b[None, :] + sp, 1e-6, 1.0)

    FAMS = ["independent-lowrate", "specialist", "near-duplicate", "partial-overlap"]
    v = {k: 0 for k in ("T1", "T2lo", "T2up", "T3id", "T3loc", "T4", "cert")}
    n_cells = 0
    for t in range(600):
        rg = np.random.default_rng(9000 + t)
        M = int(rg.integers(2, 7))
        R = draw(FAMS[t % 4], M, int(rg.integers(40, 300)), rg)
        for tau in (0.02, 0.05, 0.1, 0.2, 1 / 3, 0.5, 0.7, 0.9):
            n_cells += 1
            D = disagreement_rate(R, tau)
            L = latent_gap(R, tau)
            if L > D + 1e-12:
                v["T1"] += 1
            fl = mixture_floor(R, tau)
            if np.isfinite(fl) and cover_at_tau(mixture_rates(R), tau) < fl - 1e-12:
                v["T2lo"] += 1
            for sv in (0.0, 0.5 * tau):
                if cover_at_tau(mixture_rates(R), tau) > \
                        breadth_height_bound(R, tau, sv) + 1e-12:
                    v["T2up"] += 1
            c = mixture_certificate(R, tau)
            if c["guaranteed"] and c["realised_gain"] <= 0.0:
                v["cert"] += 1
            asg = rg.integers(0, M, R.shape[1])
            rl = routing_loss(R, asg, tau)
            if not rl["identity_holds"]:
                v["T3id"] += 1
            if rl["loss"] > rl["D_times_eD"] + 1e-9:
                v["T3loc"] += 1
            lam = rg.dirichlet(np.ones(M), size=R.shape[1]).T
            if cover_at_tau(teacher_mixture_rates(R, lam), tau) > \
                    cover_at_tau(oracle_rates(R), tau) + 1e-12:
                v["T4"] += 1
            lam_star = np.zeros_like(R)
            lam_star[np.argmax(R, axis=0), np.arange(R.shape[1])] = 1.0
            if abs(cover_at_tau(teacher_mixture_rates(R, lam_star), tau)
                   - cover_at_tau(oracle_rates(R), tau)) > 1e-12:
                v["T4"] += 1

    check(f"Thm 1 master bound L<=D ({n_cells} cells)", v["T1"] == 0, f"{v['T1']}")
    check("Thm 2 lower bound C_mix(tau) >= C_orac(M tau)", v["T2lo"] == 0, f"{v['T2lo']}")
    check("Thm 2 breadth-height upper bound", v["T2up"] == 0, f"{v['T2up']}")
    check("Cor 1 positive certificate implies a realised gain", v["cert"] == 0, f"{v['cert']}")
    check("Thm 3 routing identity loss == D * e_D", v["T3id"] == 0, f"{v['T3id']}")
    check("Thm 3 localisation bound", v["T3loc"] == 0, f"{v['T3loc']}")
    check("Thm 4 teacher-mixture ceiling and equality case", v["T4"] == 0, f"{v['T4']}")

    print("\n[2] Theorem 2 tightness: one active member gives exact equality")
    tight_ok = True
    for M in (2, 3, 4, 5):
        rg = np.random.default_rng(M)
        R = np.zeros((M, 800)); R[0] = rg.uniform(0, 1, 800)
        for tau in (0.05, 0.1, 0.15):
            if M * tau <= 1.0:
                if abs(cover_at_tau(mixture_rates(R), tau) - mixture_floor(R, tau)) > 1e-12:
                    tight_ok = False
    check("factor-M bound is exactly tight, so M cannot be improved", tight_ok)
    # the headline: uniform mixing can be strictly worse than the best member
    rg = np.random.default_rng(4)
    R = np.full((3, 4000), 1e-3)
    R[rg.integers(0, 3, 4000), np.arange(4000)] = rg.uniform(0.3, 1.0, 4000)
    cb = max(cover_at_tau(R[m], 0.30) for m in range(3))
    check("uniform mixing is strictly worse than the best member at tau=0.30",
          cover_at_tau(mixture_rates(R), 0.30) - cb < -0.05,
          f"gain {cover_at_tau(mixture_rates(R),0.30)-cb:.3f}")
    check("while the latent gap at the same tau is large and positive",
          cover_at_tau(oracle_rates(R), 0.30) - cb > 0.5,
          f"L {cover_at_tau(oracle_rates(R),0.30)-cb:.3f}")

    print("\n[2b] Optimal-mixture ceiling: no fixed weighting escapes dilution")
    ceil_viol, vertex_wins, cells = 0, 0, 0
    for M in (2, 3, 5, 8):
        for tau in (0.3, 0.5, 0.7):
            rg = np.random.default_rng(900 + 17 * M + int(100 * tau))
            sfl, n = 1e-3, 3000
            R = np.full((M, n), sfl)
            R[rg.integers(0, M, n), np.arange(n)] = rg.uniform(0.3, 1.0, n)
            cands = [np.eye(M)[i] for i in range(M)] + [np.full(M, 1.0 / M)]
            for k in range(1, M + 1):
                for _ in range(15):
                    idx = rg.choice(M, k, replace=False)
                    w = np.zeros(M); w[idx] = 1.0 / k; cands.append(w)
            for _ in range(800):
                cands.append(rg.dirichlet(np.ones(M) * 0.3))
            best = max(cover_at_tau(mixture_rates(R, w), tau) for w in cands)
            bound = min(1.0, 1.0 / (M * (tau - sfl)))
            cells += 1
            if best > bound + 1e-9:
                ceil_viol += 1
            vbest = max(cover_at_tau(R[m], tau) for m in range(M))
            if vbest >= best - 1e-9:
                vertex_wins += 1
    check(f"ceiling 1/(M(tau-s)) holds for every weighting ({cells} cells)",
          ceil_viol == 0, f"{ceil_viol}")
    check("in most cells the optimum is a vertex, i.e. mixing collapses to selection",
          vertex_wins >= cells // 2, f"{vertex_wins}/{cells}")

    print("\n[2c] Edge cases: M=1, tau extremes, ties, degenerate profiles")
    R1 = np.random.default_rng(1).uniform(0, 1, (1, 400))
    check("M=1 gives D(tau)=0 and L(tau)=0 at every tau",
          all(abs(disagreement_rate(R1, t)) < 1e-12 and abs(latent_gap(R1, t)) < 1e-12
              for t in (0.1, 0.5, 0.9)))
    Rx = np.random.default_rng(2).uniform(0, 1, (3, 2000))
    check("L <= D survives tau -> 0 and tau = 1",
          all(latent_gap(Rx, t) <= disagreement_rate(Rx, t) + 1e-12 for t in (1e-9, 1.0)))
    for nm, Rd in (("all-zero", np.zeros((3, 80))), ("all-one", np.ones((3, 80)))):
        check(f"degenerate {nm} profile satisfies L<=D and the factor-M law",
              all(latent_gap(Rd, t) <= disagreement_rate(Rd, t) + 1e-12 for t in (0.1, 0.5))
              and all(cover_at_tau(mixture_rates(Rd), t)
                      >= cover_at_tau(oracle_rates(Rd), 3 * t) - 1e-12 for t in (0.1, 0.2)))
    Rt = np.array([[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]])
    check("exact ties across members give D=L=0",
          abs(disagreement_rate(Rt, 0.5)) < 1e-12 and abs(latent_gap(Rt, 0.5)) < 1e-12)
    bad = 0
    for tr in range(300):
        rg = np.random.default_rng(4000 + tr)
        M = int(rg.integers(2, 6)); Rr = rg.beta(0.5, 3.0, (M, 300))
        for t in (0.05, 0.2, 0.5):
            D, L = disagreement_rate(Rr, t), latent_gap(Rr, t)
            if D > 1e-9 and not (-1e-12 <= L / D <= 1 + 1e-12):
                bad += 1
    check("break-even L/D lies in [0,1], which Lemma 1 guarantees", bad == 0, f"{bad}")
    bad2 = 0
    for tr in range(300):
        rg = np.random.default_rng(5000 + tr)
        M = int(rg.integers(2, 6)); Rr = rg.beta(0.6, 2.0, (M, 250))
        for t in (0.3, 0.5, 0.8, 0.95):
            if M * t > 1.0 and cover_at_tau(mixture_rates(Rr), t) < \
                    cover_at_tau(oracle_rates(Rr), min(M * t, 1.0)) - 1e-12:
                bad2 += 1
    check("factor-M law needs no restriction on tau (holds when M*tau>1)",
          bad2 == 0, f"{bad2}")

    print("\n[3] Budget invariance of coverage")
    rng = np.random.default_rng(7)
    Rr = rng.uniform(0.02, 0.95, size=(4, 400))
    r_mix = mixture_rates(Rr)
    cov_ref = restricted_auc(r_mix, 0.05, 0.60)
    check("rAUC is a pure function of the rate vector",
          all(abs(restricted_auc(r_mix, 0.05, 0.60) - cov_ref) < 1e-15 for _ in range(3)))
    pk = [pass_at_k(r_mix, k) for k in (1, 2, 8, 32, 128)]
    check("Pass@k is strictly increasing in k (contrast)",
          all(pk[i] < pk[i + 1] for i in range(len(pk) - 1)), f"{pk}")
    errs = []
    for k in (16, 64, 256, 1024):
        sc, dr = synth_success_counts(Rr, k, seed=1)
        errs.append(abs(restricted_auc(mixture_rates(sc / dr), 0.05, 0.60) - cov_ref))
    check("estimation error shrinks with k while the estimand does not move",
          errs[-1] < errs[0], f"{['%.4f' % e for e in errs]}")
    stud = np.minimum(oracle_rates(Rr) + 0.03, 1.0)
    check("a single policy is not bounded by the oracle (partial order)",
          cover_at_tau(stud, 0.97) > cover_at_tau(oracle_rates(Rr), 0.97))

    print("\n[4] Clopper-Pearson at zero successes")
    _, u32 = clopper_pearson(0, 32, alpha=0.05, side="upper")
    closed = 1.0 - 0.05 ** (1.0 / 32)
    check("0/32 one-sided 95% upper bound ~= 0.0894 as quoted",
          abs(u32 - closed) < 1e-12 and 0.085 < u32 < 0.093, f"got {u32:.6f}")
    lo, hi = clopper_pearson(3, 32, alpha=0.05)
    check("two-sided bounds bracket the point estimate", lo < 3 / 32 < hi,
          f"({lo:.4f}, {hi:.4f})")
    check("regularised incomplete beta is a CDF in x",
          _betainc_reg(2, 5, 0.0) == 0.0 and abs(_betainc_reg(2, 5, 1.0) - 1.0) < 1e-9)

    print("\n[5] Empirical-Bayes shrinkage recovers a known prior")
    a_true, b_true = 2.0, 8.0
    rg = np.random.default_rng(11)
    p_items = rg.beta(a_true, b_true, size=1500)
    s, d = synth_success_counts(p_items[None, :], 32, seed=3)
    a_hat, b_hat = eb_beta_binomial_prior(s, d)
    m_true, m_hat = a_true / (a_true + b_true), a_hat / (a_hat + b_hat)
    check("recovered prior mean is close to truth", abs(m_true - m_hat) < 0.03,
          f"true {m_true:.4f} vs fitted {m_hat:.4f}")
    post = posterior_rates(s, d, (a_hat, b_hat))
    mle = s / d
    check("posterior means are shrunk toward the prior mean",
          np.abs(post - m_hat).mean() < np.abs(mle - m_hat).mean(),
          f"post {np.abs(post-m_hat).mean():.4f} vs mle {np.abs(mle-m_hat).mean():.4f}")
    check("shrinkage reduces mean squared error vs the raw MLE",
          np.mean((post - p_items) ** 2) < np.mean((mle - p_items) ** 2))

    print("\n[6] Estimand plumbing and guards")
    base = np.full(50, 0.05)
    Rp = np.vstack([np.linspace(0.02, 0.9, 50), np.linspace(0.9, 0.02, 50)])
    g = g_multi(Rp, base, 0.05, 0.60)
    check("g_multi returns a finite gap and names its comparator",
          np.isfinite(g["g_multi"]) and g["comparator_selected_on"] == "test")
    g_val = g_multi(Rp, base, 0.05, 0.60, comparator_index=1 - g["comparator_index"])
    check("test-selected comparator is at least as strong as any other",
          g["g_multi"] <= g_val["g_multi"] + 1e-12,
          f"test {g['g_multi']:.5f} vs other {g_val['g_multi']:.5f}")
    rf = recovery_fractions(0.10, 0.06, 0.03)
    check("recovery fractions computed when the denominator is real",
          abs(rf["rho_W"] - 0.6) < 1e-12 and abs(rf["rho_S"] - 0.3) < 1e-12)
    rf0 = recovery_fractions(0.002, 0.001, 0.0, latent_gap_ci=(-0.01, 0.02))
    check("recovery fractions suppressed when the latent gap straddles zero",
          rf0["suppressed"] and rf0["rho_W"] is None)

    print("\n[7] Inference over prompts")
    pool = mixture_rates(np.vstack([base, Rp]))
    comp = mixture_rates(np.vstack([base, Rp[g["comparator_index"]]]))
    pb = paired_bootstrap_gap(pool, comp, 0.05, 0.60, n_boot=600, seed=5)
    check("paired bootstrap reports a one-sided lower bound",
          pb["ci_lo"] <= pb["gap"] <= pb["ci_hi"] and "positive_lcb" in pb)
    # NB: `pool` above is an exactly constant rate vector by construction
    # (the two parents' rates sum to a constant), so its band is degenerate.
    # That is a property of the fixture, so we assert it, then use a
    # heterogeneous vector for the substantive band check.
    band_deg = simultaneous_band(pool, np.linspace(0.05, 0.60, 25), n_boot=200, seed=5)
    check("a constant-rate curve is reported as a degenerate band, not a precise one",
          band_deg["degenerate"] is True)
    het = np.random.default_rng(21).uniform(0.0, 1.0, 300)
    band = simultaneous_band(het, np.linspace(0.05, 0.60, 25), n_boot=400, seed=5)
    check("simultaneous band contains the point curve",
          (not band["degenerate"])
          and np.all(band["lo"] <= band["point"] + 1e-12)
          and np.all(band["point"] <= band["hi"] + 1e-12))
    check("sup-t critical value exceeds the pointwise 1.96",
          band["crit"] > 1.96, f"crit {band['crit']:.3f}")
    ni_bad = non_inferiority(np.full(200, 0.30), np.full(200, 0.40), margin=0.01,
                             n_boot=400, seed=5)
    ni_ok = non_inferiority(np.full(200, 0.399), np.full(200, 0.40), margin=0.01,
                            n_boot=400, seed=5)
    check("a real accuracy loss fails non-inferiority", not ni_bad["non_inferior"])
    check("a negligible difference passes non-inferiority", ni_ok["non_inferior"])
    mask = adaptive_allocation_targets(s[:, :40], d[:, :40], np.array([0.2]))
    check("adaptive allocation flags a strict subset of items",
          0 < int(mask.sum()) <= mask.size, f"{int(mask.sum())}/{mask.size}")

    print("\n[8] Optional cross-check against scipy, if installed")
    try:
        from scipy.special import betainc as _sp_betainc
        from scipy.stats import beta as _sp_beta
        d1 = abs(_betainc_reg(2.5, 7.5, 0.3) - float(_sp_betainc(2.5, 7.5, 0.3)))
        lo_s = float(_sp_beta.ppf(0.025, 3, 32 - 3 + 1))
        hi_s = float(_sp_beta.ppf(0.975, 3 + 1, 32 - 3))
        lo_n, hi_n = clopper_pearson(3, 32, alpha=0.05)
        check("numpy incomplete beta matches scipy", d1 < 1e-6, f"delta {d1:.2e}")
        check("numpy Clopper-Pearson matches scipy",
              abs(lo_n - lo_s) < 1e-4 and abs(hi_n - hi_s) < 1e-4,
              f"numpy ({lo_n:.6f},{hi_n:.6f}) scipy ({lo_s:.6f},{hi_s:.6f})")
        n_scipy = 2
    except ImportError:
        n_scipy = 0
        print("  SKIP  scipy not installed (numpy path is authoritative)")

    # Pin the check count so it cannot drift silently. The scipy block is
    # optional, so the total is 42 where scipy is installed and 40 where it is not.
    n_core = 40
    expected = n_core + n_scipy
    print("\n" + "=" * 66)
    print(f"csm_metrics smoke test: {ok} passed, {fail} failed")
    print("=" * 66)
    if fail:
        raise SystemExit(1)
    if ok != expected:
        raise SystemExit(
            f"check count drifted: expected {expected} "
            f"({n_core} core + {n_scipy} optional scipy), got {ok}. "
            "Update n_core here.")
    print("No experimental results are produced by this file. Every number above\n"
          "is algebra from the paper, recomputed. Real rollouts must be supplied\n"
          "by the experiments before any claim is reported.")


if __name__ == "__main__":
    _smoke()
