# src/lattice_kmax/sstar.py
import numpy as np
from itertools import combinations
from .utils import pairwise_lower_bound_s


# ---------------------------
# Pair candidate (exact N = 2)
# ---------------------------
def _pair_min_scale(centers, i, j, alpha_idx):
    """
    Minimal scale s for two spheres i,j to touch on the surface:
        ||ci - cj|| = s (alpha_i + alpha_j)
    """
    d = np.linalg.norm(centers[i] - centers[j])
    return d / (alpha_idx[i] + alpha_idx[j])


# ---------------------------------------
# Triple candidate (exact N = 3), general
# ---------------------------------------
def _triple_min_scale(centers, idxs, alpha_idx, tol=1e-14):
    """
    Minimal s such that there exists x with |x-ci| = s*alpha_i for i in the triple.
    Works for arbitrary fixed ratios alpha_idx.
    """
    i1, i2, i3 = idxs
    c1, c2, c3 = centers[i1], centers[i2], centers[i3]
    a1, a2, a3 = alpha_idx[i1], alpha_idx[i2], alpha_idx[i3]

    A = 2.0 * np.vstack([c2 - c1, c3 - c1])  # 2×3
    if np.linalg.matrix_rank(A) < 2:
        return None

    b0 = np.array([np.dot(c2, c2) - np.dot(c1, c1),
                   np.dot(c3, c3) - np.dot(c1, c1)])
    d = np.array([a2*a2 - a1*a1, a3*a3 - a1*a1])

    # Pseudoinverse and nullspace
    Ap = np.linalg.pinv(A, rcond=1e-12)
    U, S, Vt = np.linalg.svd(A)
    n = Vt.T[:, -1]  # A@n ≈ 0

    X0 = Ap @ b0
    X1 = -Ap @ d

    beta0 = np.dot(X0 - c1, n)
    beta1 = np.dot(X1, n)
    c0 = np.dot(X0, X0) - 2*np.dot(c1, X0) + np.dot(c1, c1)
    c1coef = 2*(np.dot(X0, X1) - np.dot(c1, X1)) - a1*a1
    c2 = np.dot(X1, X1)

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
    P = np.vstack([centers[i] for i in idxs])    # 4×3
    al = np.array([alpha_idx[i] for i in idxs])  # 4
    c0 = P[0]
    A = 2.0 * (P[1:] - c0)                       # 3×3
    if np.linalg.matrix_rank(A) < 3:
        return None
    b0 = np.sum(P[1:]**2, axis=1) - np.sum(c0**2)
    d = (al[1:]**2 - al[0]**2)
    Ainv = np.linalg.inv(A)

    x0 = Ainv @ b0
    x1 = -Ainv @ d

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
def s_star_fixed_ratios(centers, alpha_idx, neighbor_index, N_max=6,
                        eps=1e-8, kNN=32):
    """
    Returns dict {N: s_N_star} for N=1..N_max.
    Uses k-NN neighbourhoods for all N>=2 to avoid cutoff artefacts.
    """
    s_star = {N: np.inf for N in range(1, N_max + 1)}
    s_star[1] = 0.0

    n = len(centers)
    if n <= 1:
        return s_star

    diffs = centers[1:] - centers[0]
    a_char = float(np.min(np.linalg.norm(diffs, axis=1))) if len(diffs) else 1.0
    local_r = 2.5 * a_char

    # --- k-NN query ---
    kNN_eff = min(max(1, kNN), max(1, n - 1))
    try:
        dists, idxs = neighbor_index.tree.query(centers, k=kNN_eff + 1)
    except Exception:
        D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        idxs = np.argsort(D, axis=1)[:, :kNN_eff + 1]

    # -------- Pairs (N=2) --------
    s2 = np.inf
    for i in range(n):
        for j in idxs[i][1:]:
            if j <= i:
                continue
            s2 = min(s2, _pair_min_scale(centers, i, j, alpha_idx))
    if 2 <= N_max and s2 < s_star[2]:
        s_star[2] = s2

    # -------- Triples (N=3) --------
    s3 = np.inf
    for i in range(n):
        nbr = [int(j) for j in idxs[i][1:]]
        for j, k in combinations(nbr, 2):
            if j == i or k == i or j == k:
                continue
            out = _triple_min_scale(centers, (i, j, k), alpha_idx)
            if out is not None and out < s3:
                s3 = out
    if 3 <= N_max and s3 < s_star[3]:
        s_star[3] = s3
        if s3 < s_star.get(2, np.inf):
            s_star[2] = s3

    # -------- Quadruples (N>=4) --------
    seen = set()
    for i in range(n):
        nbr = [int(j) for j in idxs[i][1:]]
        for j, k, l in combinations(nbr, 3):
            key = tuple(sorted((i, j, k, l)))
            if key in seen:
                continue
            seen.add(key)

            idxs4 = (i, j, k, l)
            al4 = np.array([alpha_idx[t] for t in idxs4])

            lb = pairwise_lower_bound_s(centers, idxs4, al4)
            if lb >= s_star.get(N_max, np.inf):
                continue

            out = _quadruple_s_and_center(centers, idxs4, alpha_idx)
            if out is None:
                continue
            s, x = out
            if s >= s_star.get(N_max, np.inf):
                continue

            loc = neighbor_index.local_around_point(x, local_r)
            d = np.linalg.norm(centers[loc] - x, axis=1)
            targ = s * np.asarray([alpha_idx[t] for t in loc])
            m = int(np.sum(np.abs(d - targ) <= eps))

            if m >= 1:
                for N in range(1, min(N_max, m) + 1):
                    if s < s_star[N]:
                        s_star[N] = s

    return s_star
