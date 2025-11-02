# src/lattice_kmax/sstar.py
import numpy as np
from itertools import combinations
from .utils import pairwise_lower_bound_s

# ---------------------------
# Pair candidate (exact N = 2)
# ---------------------------
def _pair_min_scale(centers, i, j, alpha_idx):
    d = np.linalg.norm(centers[i] - centers[j])
    return d / (alpha_idx[i] + alpha_idx[j])


# ---------------------------------------
# Triple candidate (exact N = 3), general
# ---------------------------------------
def _triple_min_scale(centers, idxs, alpha_idx, tol=1e-14):
    """
    Minimal s such that there exists x with |x-ci| = s*alpha_i for i in the triple.
    Works for arbitrary (fixed) ratios alpha_idx.
    Returns s (float) or None if degenerate.
    """
    i1, i2, i3 = idxs
    c1, c2, c3 = centers[i1], centers[i2], centers[i3]
    a1, a2, a3 = alpha_idx[i1], alpha_idx[i2], alpha_idx[i3]

    # Two-plane system from subtracting sphere equations (unknowns x in R^3):
    # 2(c2-c1)·x = |c2|^2 - |c1|^2 + s^2(a2^2 - a1^2)
    # 2(c3-c1)·x = |c3|^2 - |c1|^2 + s^2(a3^2 - a1^2)
    A = 2.0 * np.vstack([c2 - c1, c3 - c1])          # 2x3
    if np.linalg.matrix_rank(A) < 2:
        return None  # collinear / degenerate

    b0 = np.array([np.dot(c2, c2) - np.dot(c1, c1),
                   np.dot(c3, c3) - np.dot(c1, c1)])   # 2
    d  = np.array([a2*a2 - a1*a1, a3*a3 - a1*a1])      # 2

    # Pseudoinverse & nullspace of A
    U, S, Vt = np.linalg.svd(A)
    Sinv = np.zeros_like(S)
    nz = S > 1e-14
    Sinv[nz] = 1.0 / S[nz]
    Ap = Vt.T @ np.diag(Sinv) @ U.T     # 3x2 pseudoinverse
    n  = Vt.T[:, -1]                    # unit nullspace vector, A n = 0

    # x(λ, t) = X0 + λ X1 + t n, with λ = s^2
    X0 = Ap @ b0
    X1 = -Ap @ d

    # Impose sphere-1: |x|^2 - 2 c1·x + |c1|^2 = λ a1^2
    # Substitute x = X0 + λ X1 + t n; choose minimal λ by requiring
    # the line in t to be tangent → discriminant = 0:
    beta0 = np.dot(X0 - c1, n)
    beta1 = np.dot(X1, n)
    c0 = np.dot(X0, X0) - 2*np.dot(c1, X0) + np.dot(c1, c1)
    c1coef = 2*(np.dot(X0, X1) - np.dot(c1, X1)) - a1*a1
    c2 = np.dot(X1, X1)

    # Quadratic in λ: (beta1^2 - c2) λ^2 + (2 beta0 beta1 - c1coef) λ + (beta0^2 - c0) = 0
    A2 = beta1*beta1 - c2
    B2 = 2*beta0*beta1 - c1coef
    C2 = beta0*beta0 - c0

    if abs(A2) < tol:
        if abs(B2) < tol:
            return None
        lam = -C2 / B2
        if lam < 0:
            return None
    else:
        disc = B2*B2 - 4*A2*C2
        if disc < 0:
            return None
        r1 = (-B2 - np.sqrt(disc)) / (2*A2)
        r2 = (-B2 + np.sqrt(disc)) / (2*A2)
        cands = [r for r in (r1, r2) if r >= 0]
        if not cands:
            return None
        lam = min(cands)

    return float(np.sqrt(lam))


# -------------------------------------------------------
# Quadruple candidate (N >= 4): Apollonius / orthosphere
# -------------------------------------------------------
def _quadruple_s_and_center(centers, idxs, alpha_idx, tol=1e-16):
    """
    Solve for (s, x) where four spheres (fixed ratios) are tangent at x.
    Returns (s, x) or None.
    """
    P = np.vstack([centers[i] for i in idxs])    # 4x3
    al = np.array([alpha_idx[i] for i in idxs])  # 4
    c0 = P[0]
    A = 2.0 * (P[1:] - c0)                       # 3x3
    if np.linalg.matrix_rank(A) < 3:
        return None
    b0 = np.sum(P[1:]**2, axis=1) - np.sum(c0**2)
    d  = (al[1:]**2 - al[0]**2)
    Ainv = np.linalg.inv(A)

    # x(λ) = x0 + λ x1, with λ = s^2
    x0 = Ainv @ b0
    x1 = -Ainv @ d

    # Tangency on sphere 0: |x|^2 - 2 c0·x + |c0|^2 - λ al0^2 = 0
    a = np.dot(x1, x1)
    b = 2*np.dot(x0, x1) - 2*np.dot(c0, x1) - (al[0]**2)
    c = np.dot(x0, x0) - 2*np.dot(c0, x0) + np.dot(c0, c0)

    if abs(a) < tol:
        if abs(b) < tol:
            return None
        lam = -c / b
        if lam < 0:
            return None
    else:
        disc = b*b - 4*a*c
        if disc < 0:
            return None
        r1 = (-b - np.sqrt(disc)) / (2*a)
        r2 = (-b + np.sqrt(disc)) / (2*a)
        cands = [r for r in (r1, r2) if r >= 0]
        if not cands:
            return None
        lam = min(cands)

    s = float(np.sqrt(lam))
    x = x0 + lam * x1
    return s, x


# --------------------------------------------------------
# Public: minimal s for N-way surface intersection (fixed ratios)
# --------------------------------------------------------
def s_star_fixed_ratios(centers, alpha_idx, neighbor_index, N_max=6, eps=1e-8):
    """
    centers: (n,3) float array of sphere centers
    alpha_idx: (n,) float array of per-center ratio alpha (mapped by lattice)
    neighbor_index: NeighborIndex (KD-tree wrapper)
    Returns dict {N: s_N_star} for N = 1..N_max (np.inf if none found in window).
    """
    s_star = {N: np.inf for N in range(1, N_max + 1)}
    s_star[1] = 0.0  # trivial

    n = len(centers)

    # Characteristic spacing to set a small local radius for multiplicity counting
    if n > 1:
        diffs = centers[1:] - centers[0]
        a_char = np.min(np.linalg.norm(diffs, axis=1))
    else:
        a_char = 1.0
    local_r = 2.5 * a_char

    # --------- Pairs (N = 2) ----------
    s2 = np.inf
    for i in range(n):
        for j in neighbor_index.list_for(i):
            if j <= i:
                continue
            s2 = min(s2, _pair_min_scale(centers, i, j, alpha_idx))
    if 2 <= N_max and s2 < s_star[2]:
        s_star[2] = s2

    # --------- Triples (N = 3) --------
    s3 = np.inf
    for i in range(n):
        nbr = [j for j in neighbor_index.list_for(i) if j > i]
        for j, k in combinations(nbr, 2):
            out = _triple_min_scale(centers, (i, j, k), alpha_idx)
            if out is not None:
                if out < s3:
                    s3 = out
    if 3 <= N_max and s3 < s_star[3]:
        s_star[3] = s3
        # A true triple implies at least a pair; tighten s2 if needed
        if s3 < s_star.get(2, np.inf):
            s_star[2] = s3

    # ----- Quadruples and multiplicity counting (N >= 4) -----
    # Early pruning uses current best for N_max
    for i in range(n):
        nbr = [j for j in neighbor_index.list_for(i) if j > i]
        for j, k, l in combinations(nbr, 3):
            idxs = (i, j, k, l)
            al = np.array([alpha_idx[t] for t in idxs])

            # Pairwise LB on s for this quadruple
            lb = pairwise_lower_bound_s(centers, idxs, al)
            if lb >= s_star.get(N_max, np.inf):
                continue

            out = _quadruple_s_and_center(centers, idxs, alpha_idx)
            if out is None:
                continue
            s, x = out
            # Prune if already worse than best N_max
            if s >= s_star.get(N_max, np.inf):
                continue

            # Multiplicity at (x, s): count all centers on surface within eps
            loc = neighbor_index.local_around_point(x, local_r)
            d = np.linalg.norm(centers[loc] - x, axis=1)
            targ = s * np.asarray([alpha_idx[t] for t in loc])
            m = int(np.sum(np.abs(d - targ) <= eps))

            if m >= 1:
                for N in range(1, min(N_max, m) + 1):
                    if s < s_star[N]:
                        s_star[N] = s

    return s_star
