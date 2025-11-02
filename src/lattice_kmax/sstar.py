# lattice_kmax/sstar.py
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
    Works for arbitrary fixed ratios alpha_idx. Returns s (float) or None if degenerate.
    """
    i1, i2, i3 = idxs
    c1, c2, c3 = centers[i1], centers[i2], centers[i3]
    a1, a2, a3 = alpha_idx[i1], alpha_idx[i2], alpha_idx[i3]

    # Two-plane system from subtracting sphere equations: A x = b0 - λ d, λ = s^2
    # Using absolute coordinates for x
    A = 2.0 * np.vstack([c2 - c1, c3 - c1])  # 2x3
    if np.linalg.matrix_rank(A) < 2:
        return None  # collinear / degenerate triple

    b0 = np.array([np.dot(c2, c2) - np.dot(c1, c1),
                   np.dot(c3, c3) - np.dot(c1, c1)])
    d = np.array([a2*a2 - a1*a1, a3*a3 - a1*a1])

    # Moore–Penrose pseudoinverse (3x2) and nullspace
    Ap = np.linalg.pinv(A, rcond=1e-12)
    U, S, Vt = np.linalg.svd(A)
    n = Vt.T[:, -1]  # A @ n ≈ 0

    # x(λ, t) = X0 + λ X1 + t n
    X0 = Ap @ b0
    X1 = -Ap @ d

    # Sphere-1: |x - c1|^2 = λ a1^2 ; with x = X0 + λ X1 + t n (absolute coords)
    x0m = X0 - c1
    a = np.dot(n, n)
    b = 2.0 * np.dot(x0m, n) + 2.0 * (np.dot(X1, n))  # but X1 multiplies λ; handled via quadratic below
    # Derive quadratic in λ by requiring tangency (discriminant=0) after eliminating t:
    # Line-sphere distance minimal at t* = - (x0m + λ X1)·n / |n|^2 ; plug back → quadratic in λ.
    nn = a
    pn = np.dot(X1, n) / nn
    qn = np.dot(x0m, n) / nn

    # Distance^2 at optimal t: ||x0m + λ X1||^2 - nn*(qn + λ pn)^2
    A2 = np.dot(X1, X1) - nn * (pn ** 2)
    B2 = 2.0 * (np.dot(x0m, X1) - nn * qn * pn) - a1*a1
    C2 = np.dot(x0m, x0m) - nn * (qn ** 2)

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
    d = (al[1:]**2 - al[0]**2)
    Ainv = np.linalg.inv(A)

    # x(λ) = x0 + λ x1, with λ = s^2
    x0 = Ainv @ b0
    x1 = -Ainv @ d

    # Tangency on sphere 0: |x - c0|^2 - λ al0^2 = 0
    x0m = x0 - c0
    a = np.dot(x1, x1)
    b = 2*np.dot(x0m, x1) - (al[0]**2)
    c = np.dot(x0m, x0m)

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


# -------------------------------------------------------
# Triplet-seed intersections at a fixed s (robust post-pass)
# -------------------------------------------------------
def _three_sphere_intersections(c1, r1, c2, r2, c3, r3, eps=1e-12):
    """
    Return 0/1/2 intersection points of three spheres in absolute coordinates.
    Robust near tangency.
    """
    # Build two planes: from |x-c1|^2 = r1^2 & |x-c2|^2 = r2^2, etc.
    A = np.vstack([2*(c2 - c1), 2*(c3 - c1)])  # 2x3
    b = np.array([r1*r1 - r2*r2 + np.dot(c2, c2) - np.dot(c1, c1),
                  r1*r1 - r3*r3 + np.dot(c3, c3) - np.dot(c1, c1)])

    # Pseudoinverse gives minimal-norm solution x0 to A x = b, and nullspace dir n (A n ≈ 0)
    Ap = np.linalg.pinv(A, rcond=1e-12)        # 3x2
    x0 = Ap @ b                                 # absolute coords
    U, S, Vt = np.linalg.svd(A)
    n = Vt.T[:, -1]                             # line direction

    # Intersect line x(t) = x0 + t n with sphere 1: |x - c1|^2 = r1^2
    x0m = x0 - c1
    a = np.dot(n, n)
    bq = 2*np.dot(x0m, n)
    cq = np.dot(x0m, x0m) - r1*r1
    disc = bq*bq - 4*a*cq
    if disc < -eps:
        return []
    if abs(disc) <= eps:
        t = -bq/(2*a)
        return [x0 + t*n]
    rt = np.sqrt(max(0.0, disc))
    t1, t2 = (-bq - rt)/(2*a), (-bq + rt)/(2*a)
    return [x0 + t1*n, x0 + t2*n]


def _postpass_check_at_s(centers, alpha_idx, neighbor_index, s, local_r, eps, kNN):
    """
    For a given scale s, generate candidate intersection points by 3-sphere
    intersections from k-NN neighborhoods, and return the maximum multiplicity m.
    """
    if not np.isfinite(s) or s <= 0:
        return 1
    radii = s * alpha_idx
    n = len(centers)

    # k-NN indices (include self)
    kNN_eff = min(max(1, kNN), max(1, n - 1))
    try:
        _, idxs = neighbor_index.tree.query(centers, k=kNN_eff + 1)
    except Exception:
        D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        idxs = np.argsort(D, axis=1)[:, :kNN_eff + 1]

    mmax = 1
    seen_pts = []  # de-dup by proximity
    for i in range(n):
        nbr = [int(j) for j in idxs[i][1:]]
        for j, k in combinations(nbr, 2):
            pts = _three_sphere_intersections(
                centers[i], radii[i],
                centers[j], radii[j],
                centers[k], radii[k],
                eps=1e-12
            )
            for x in pts:
                if any(np.linalg.norm(x - y) < 1e-8 for y in seen_pts):
                    continue
                seen_pts.append(x)
                loc = neighbor_index.local_around_point(x, local_r)
                d = np.linalg.norm(centers[loc] - x, axis=1)
                targ = radii[loc]
                err = np.abs(d - targ)
                thr = eps + 1e-6 * np.maximum(1.0, targ)  # relative + absolute
                m = int(np.sum(err <= thr))
                if m > mmax:
                    mmax = m
    return mmax


# --------------------------------------------------------
# Public: minimal s for N-way surface intersection (fixed ratios)
# --------------------------------------------------------
def s_star_fixed_ratios(centers, alpha_idx, neighbor_index, N_max=6,
                        eps=1e-8, kNN=32):
    """
    Returns dict {N: s_N_star} for N=1..N_max (np.inf if not found in window).
    Uses k-NN neighbourhoods for N>=2 to avoid cutoff artefacts at window edges.
    Also validates at candidate s values using triplet-seed intersections.
    """
    s_star = {N: np.inf for N in range(1, N_max + 1)}
    s_star[1] = 0.0

    n = len(centers)
    if n <= 1:
        return s_star

    # Characteristic spacing for local multiplicity search radius
    diffs = centers[1:] - centers[0]
    a_char = float(np.min(np.linalg.norm(diffs, axis=1))) if len(diffs) else 1.0
    local_r = 3.0 * a_char  # robust in mixed-α cases

    # --- k-NN query (includes self in column 0) ---
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
    quad_s_candidates = []  # collect s from quadruples for post-pass validation
    for i in range(n):
        nbr = [int(j) for j in idxs[i][1:]]
        for j, k, l in combinations(nbr, 3):
            key = tuple(sorted((i, j, k, l)))
            if key in seen:
                continue
            seen.add(key)

            idxs4 = (i, j, k, l)
            al4 = np.array([alpha_idx[t] for t in idxs4])

            # Pairwise LB on s for this quadruple
            lb = pairwise_lower_bound_s(centers, idxs4, al4)
            if lb >= s_star.get(N_max, np.inf):
                continue

            out = _quadruple_s_and_center(centers, idxs4, alpha_idx)
            if out is None:
                continue
            s, x = out
            quad_s_candidates.append(s)
            if s >= s_star.get(N_max, np.inf):
                continue

            # Multiplicity at (x, s): robust relative + absolute tolerance
            loc = neighbor_index.local_around_point(x, local_r)
            d = np.linalg.norm(centers[loc] - x, axis=1)
            targ = s * np.asarray([alpha_idx[t] for t in loc])
            err = np.abs(d - targ)
            thr = eps + 1e-6 * np.maximum(1.0, targ)
            m = int(np.sum(err <= thr))

            if m >= 1:
                for N in range(1, min(N_max, m) + 1):
                    if s < s_star[N]:
                        s_star[N] = s

    # -------- Post-pass multiplicity checks at candidate s --------
    cand_s = set()
    if np.isfinite(s2): cand_s.add(float(s2))
    if np.isfinite(s3): cand_s.add(float(s3))
    for s in quad_s_candidates:
        if np.isfinite(s):
            cand_s.add(float(s))
    for s in sorted(cand_s):
        mmax = _postpass_check_at_s(centers, alpha_idx, neighbor_index, s, local_r, eps, kNN_eff)
        if mmax >= 1:
            for N in range(1, min(N_max, mmax) + 1):
                if s < s_star[N]:
                    s_star[N] = s

    return s_star
