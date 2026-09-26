"""Text router families (linear, gradient-boosted, nearest-neighbor), fitted without test labels."""
from __future__ import annotations
import sys
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingRegressor
    HAVE_SK = True
except Exception:                                     # pragma: no cover
    HAVE_SK = False

ROUTERS = ("best_single", "tfidf_logreg", "gbm_percorrect", "knn_valrate", "oracle")


def _numpy_logreg(X, y, n_cls, iters=400, lr=0.5, l2=1e-3, seed=0):
    """Multinomial logistic regression, plain numpy. Used when sklearn is absent."""
    rng = np.random.default_rng(seed)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    W = rng.normal(0, 0.01, (X.shape[1], n_cls))
    Y = np.eye(n_cls)[y]
    for _ in range(iters):
        Z = X @ W
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        W -= lr * (X.T @ (P - Y) / X.shape[0] + l2 * W)
    return lambda Xn: np.argmax(np.hstack([Xn, np.ones((Xn.shape[0], 1))]) @ W, axis=1)


def fit_router(kind, *, train_text=None, train_feats=None, train_rates=None,
               apply_text=None, apply_feats=None, apply_rates=None, seed=0):
    """Return an assignment array over the prompts described by the apply_* inputs.

    train_rates : (M, n_train) member success rates on the TEACHER split (labels are
                  allowed here; the teacher split is never reported on).
    apply_rates : only used by 'oracle' and 'best_single'.
    """
    if kind not in ROUTERS:
        raise ValueError(f"unknown router '{kind}'; choose from {ROUTERS}")
    if kind == "oracle":
        if apply_rates is None:
            raise ValueError("'oracle' needs apply_rates and is diagnostic only")
        return np.argmax(np.atleast_2d(apply_rates), axis=0)
    if kind == "best_single":
        if train_rates is None:
            raise ValueError("'best_single' needs train_rates to pick the member")
        m = int(np.argmax(np.atleast_2d(train_rates).mean(axis=1)))
        n = (len(apply_text) if apply_text is not None
             else (apply_feats.shape[0] if apply_feats is not None
                   else np.atleast_2d(apply_rates).shape[1]))
        return np.full(n, m, dtype=int)

    R = np.atleast_2d(train_rates)
    y = np.argmax(R, axis=0)                       # target: best member on the teacher split
    Mn = R.shape[0]

    if kind == "tfidf_logreg":
        if train_text is None or apply_text is None:
            raise ValueError("'tfidf_logreg' needs train_text and apply_text")
        if HAVE_SK:
            vec = TfidfVectorizer(min_df=1, max_features=4000, ngram_range=(1, 2))
            Xtr = vec.fit_transform(train_text)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(Xtr, y)
            return clf.predict(vec.transform(apply_text)).astype(int)
        # numpy fallback: hashed bag of words
        def feat(texts, dim=512):
            F = np.zeros((len(texts), dim))
            for i, t in enumerate(texts):
                for w in str(t).lower().split():
                    F[i, hash(w) % dim] += 1.0
            n = np.linalg.norm(F, axis=1, keepdims=True); n[n == 0] = 1
            return F / n
        pred = _numpy_logreg(feat(train_text), y, Mn, seed=seed)
        return pred(feat(apply_text)).astype(int)

    if kind == "knn_valrate":
        if train_text is None or apply_text is None:
            raise ValueError("'knn_valrate' needs train_text and apply_text")
        K = 15
        if HAVE_SK:
            vec = TfidfVectorizer(min_df=1, max_features=4000, ngram_range=(1, 2))
            Xtr = vec.fit_transform(train_text); Xap = vec.transform(apply_text)
            S = (Xap @ Xtr.T).toarray()                     # cosine: tf-idf rows are l2-normalised
        else:
            def feat(texts, dim=512):
                F = np.zeros((len(texts), dim))
                for i, t in enumerate(texts):
                    for w in str(t).lower().split():
                        F[i, hash(w) % dim] += 1.0
                n = np.linalg.norm(F, axis=1, keepdims=True); n[n == 0] = 1
                return F / n
            S = feat(apply_text) @ feat(train_text).T
        nb = np.argsort(-S, axis=1)[:, :K]                  # (n_apply, K) neighbour indices
        meanR = R[:, nb].mean(axis=2)                       # (M, n_apply)
        return np.argmax(meanR, axis=0)

    # gbm_percorrect: predict each member's rate, then argmax
    if train_feats is None or apply_feats is None:
        raise ValueError("'gbm_percorrect' needs train_feats and apply_feats")
    Xtr, Xap = np.asarray(train_feats, float), np.asarray(apply_feats, float)
    P = np.zeros((Mn, Xap.shape[0]))
    for m in range(Mn):
        if HAVE_SK:
            g = GradientBoostingRegressor(n_estimators=120, max_depth=3,
                                          random_state=seed)
            g.fit(Xtr, R[m]); P[m] = g.predict(Xap)
        else:                                       # ridge fallback
            A = np.hstack([Xtr, np.ones((Xtr.shape[0], 1))])
            w = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ R[m])
            P[m] = np.hstack([Xap, np.ones((Xap.shape[0], 1))]) @ w
    return np.argmax(P, axis=0)


def _selftest() -> int:
    import csm_metrics as MM
    rng = np.random.default_rng(0)
    n_tr, n_ap, Mn, tau = 600, 400, 3, 0.3
    # a learnable signal: a keyword in the prompt names the competent member
    words = ["chart", "geometry", "scene"]
    own_tr = rng.integers(0, Mn, n_tr); own_ap = rng.integers(0, Mn, n_ap)
    txt_tr = [f"solve the {words[o]} question number {i}" for i, o in enumerate(own_tr)]
    txt_ap = [f"solve the {words[o]} question number {i}" for i, o in enumerate(own_ap)]
    def rates(own, n):
        R = rng.beta(0.5, 12.0, (Mn, n))
        R[own, np.arange(n)] = rng.uniform(0.35, 0.95, n)
        return R
    Rtr, Rap = rates(own_tr, n_tr), rates(own_ap, n_ap)
    Ftr = np.eye(Mn)[own_tr] + rng.normal(0, 0.25, (n_tr, Mn))
    Fap = np.eye(Mn)[own_ap] + rng.normal(0, 0.25, (n_ap, Mn))

    print(f"sklearn available: {HAVE_SK}")
    print(f"{'router':16s} {'Cover@tau':>10s} {'e_D':>7s} {'needs<':>7s} {'beats best':>11s}")
    ok = True
    base = MM.cover_at_tau(Rap[int(np.argmax(Rtr.mean(1)))], tau)
    orac = MM.cover_at_tau(MM.oracle_rates(Rap), tau)
    for k in ROUTERS:
        asg = fit_router(k, train_text=txt_tr, train_feats=Ftr, train_rates=Rtr,
                         apply_text=txt_ap, apply_feats=Fap, apply_rates=Rap)
        if len(asg) != n_ap:
            print(f"FAIL {k}: assignment length {len(asg)} != {n_ap}"); ok = False; continue
        r = MM.routing_loss(Rap, asg, tau)
        cov = MM.cover_at_tau(MM.router_rates(Rap, asg), tau)
        if cov > orac + 1e-9:
            print(f"FAIL {k}: exceeds the oracle, which Theorem 5 forbids"); ok = False
        print(f"{k:16s} {cov:10.4f} {r['e_D']:7.3f} {r['break_even_accuracy']:7.3f}"
              f" {str(bool(cov > base + 1e-12)):>11s}")
    print(f"\nreference: best single {base:.4f}, oracle {orac:.4f}")
    print("all four families produce a valid assignment and none exceeds the oracle"
          if ok else "SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest() if len(sys.argv) < 2 or sys.argv[1] == "selftest" else 0)
