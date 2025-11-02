# lattice_kmax/sstar.py
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
    i1, i2, i3 = idxs
    c1, c2, c3 = centers[i1], centers[i2], centers[i3]
    a1, a2, a3 = alpha_idx[i1], alpha_idx[i2], alpha_idx[i3]

    A = 2.0 * np.vstack([c2 - c1, c3 - c1])  # 2x3
    if np.linalg.matrix_rank(A) < 2:
        return None

    b0 = np.array([np.dot(c2, c2) - np.dot(c1, c1),
                   np.dot(c3, c3) - np.dot(c1, c1)])
    d = np.array([a2*a2 - a1*a1, a3*a3 - a1*a1])

    Ap = np.linalg.pinv(A, rcond=1e-12)       # 3x2
    U, S, Vt = np.linalg.svd(A)
    n = Vt.T[:, -1]                           # A@n ≈ 0

    X0 = Ap @ b0
    X1 = -Ap @ d

    # Line-sphere distance minimisation (absolute coords):
    x0m = X0 - c1
    nn = np.dot(n, n)
    pn = np.dot(X1, n) / nn
    qn = np.dot(x0m, n) / nn

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
    P = np.vstack([centers[i] for i in idxs])    # 4x3
    al = np.array([alpha_idx[i] for i in idxs])  # 4
    c0 = P[0]
    A = 2.0 * (P[1:] - c0)                       # 3x3
    if np.linalg.matrix_rank(A) < 3:
        return None
    b0 = np.sum(P[1:]**2, axis=1) - np.sum(c0**2)
    d = (al[1:]**2 - al[0]**2)
    Ainv = np.linalg.inv(A)

    x0 = Ainv @ b0
    x1 = -Ainv @ d

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
    Return 0/1/2 intersection points of three spheres (absolute coords).
    """
    A = np.vstack([2*(c2 - c1), 2*(c3 - c1)])  # 2x3
    b = np.array([
        r1*r1 - r2*r2 + np.dot(c2, c2) - np.dot(c1, c1),
        r1*r1 - r3*r3 + np.dot(c3, c3) - np.dot(c1, c1),
    ])

    Ap = np.linalg.pinv(A, rcond=1e-12)        # 3x2
    x0 = Ap @ b
    U, S, Vt = np.linalg.svd(A)
    if (S > 1e-12).sum() < 2:
        return []
    n = Vt.T[:, -1]                             # line direction

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


def _postpass_check_at_s(centers, alpha_idx, neighbor_index, s,
                         local_r, eps,
                         postpass_triplet_k=12, max_triplet_tests=4000,
                         early_stop_N=None):
    """
    For a given scale s, generate candidate intersection points by 3-sphere
    intersections from a reduced k-NN and return the maximum multiplicity m.
    Limits work via postpass_triplet_k and max_triplet_tests.
    """
    if not np.isfinite(s) or s <= 0:
        return 1
    radii = s * alpha_idx
    n = len(centers)

    # Smaller k just for the post-pass to bound combinations
    k_eff = min(max(1, postpass_triplet_k), max(1, n - 1))
    try:
        _, idxs = neighbor_index.tree.query(centers, k=k_eff + 1)
    except Exception:
        D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        idxs = np.argsort(D, axis=1)[:, :k_eff + 1]

    mmax = 1
    seen_pts = []
    tested = 0
    for i in range(n):
        nbr = [int(j) for j in idxs[i][1:]]
        for j, k in combinations(nbr, 2):
            pts = _three_sphere_intersections(
                centers[i], radii[i], centers[j], radii[j], centers[k], radii[k]
            )
            tested += 1
            for x in pts:
                # de-dup tested points
                if any(np.linalg.norm(x - y) < 1e-8 for y in seen_pts):
                    continue
                seen_pts.append(x)

                loc = neighbor_index.local_around_point(x, local_r)
                d = np.linalg.norm(centers[loc] - x, axis=1)
                targ = radii[loc]
                err = np.abs(d - targ)
                thr = eps + 1e-6 * np.maximum(1.0, targ)
                m = int(np.sum(err <= thr))
                if m > mmax:
                    mmax = m
                    # early stop if we've already matched the current best need
                    if early_stop_N is not None and mmax >= early_stop_N:
                        return mmax
            if tested >= max_triplet_tests:
                return mmax
    return mmax


# --------------------------------------------------------
# Public: minimal s for N-way surface intersection (fixed ratios)
# --------------------------------------------------------
def s_star_fixed_ratios(centers, alpha_idx, neighbor_index, N_max=6,
                        eps=1e-8, kNN=32,
                        max_postpass_scales=16,   # cap number of scales to validate
                        postpass_triplet_k=12,    # smaller k for post-pass
                        max_triplet_tests=4000):  # cap triplet checks per s
    """
    Returns dict {N: s_N_star} for N=1..N_max (np.inf if not found in window).
    Uses k-NN neighbourhoods for N>=2; validates a bounded set of scales via fast post-pass.
    """
    s_star = {N: np.inf for N in range(1, N_max + 1)}
    s_star[1] = 0.0

    n = len(centers)
    if n <= 1:
        return s_star

    # Characteristic spacing for local multiplicity search radius
    diffs = centers[1:] - centers[0]
    a_char = float(np.min(np.linalg.norm(diffs, axis=1))) if len(diffs) else 1.0
    local_r = 3.0 * a_char

    # --- k-NN for main search (includes self in column 0) ---
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
    quad_s_candidates = []
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
            quad_s_candidates.append(float(s))
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

    # -------- Post-pass multiplicity checks at a bounded set of s --------
    # Build candidate s set: {s2, s3} U smallest M from quads, dedup within rel tol.
    cand = []
    if np.isfinite(s2): cand.append(float(s2))
    if np.isfinite(s3): cand.append(float(s3))
    quad_s_candidates.sort()
    for s in quad_s_candidates[:max_postpass_scales]:
        cand.append(s)

    # Deduplicate (relative tol 1e-8)
    cand_sorted = []
    for s in sorted(cand):
        if not cand_sorted or abs(s - cand_sorted[-1]) > 1e-8 * max(1.0, cand_sorted[-1]):
            cand_sorted.append(s)

    # Post-check each s with small k and capped triplet count; early stop when possible
    for s in cand_sorted:
        # If we already have all N up to N_max at <= s, we could skip; simple guard:
        # nothing fancy—just run the check and update.
        mmax = _postpass_check_at_s(
            centers, alpha_idx, neighbor_index, s, local_r, eps,
            postpass_triplet_k=postpass_triplet_k,
            max_triplet_tests=max_triplet_tests,
            early_stop_N=N_max
        )
        if mmax >= 1:
            for N in range(1, min(N_max, mmax) + 1):
                if s < s_star[N]:
                    s_star[N] = s

    return s_star
