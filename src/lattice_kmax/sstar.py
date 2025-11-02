# src/lattice_kmax/sstar.py
# v0.2.0 — quad solve + fast pair-circle validator
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
# Fast validator: pair-circle sampling (visual-style)
# -------------------------------------------------------
def _fast_visual_validator_at_s(centers, alpha_idx, neighbor_index, s,
                                a_char,
                                samples_per_circle=8,
                                tol_abs=1e-3,
                                tol_rel=1e-6,
                                max_pairs_per_site=48,
                                early_stop_N=None):
    """
    Fast approximate multiplicity check at fixed scale s.
    Returns m_max detected at this s.
    """
    if not np.isfinite(s) or s <= 0:
        return 1
    n = len(centers)
    if n == 0:
        return 1

    radii = s * alpha_idx

    # k-NN limited neighbor list per site to cap pairs
    k = min(max_pairs_per_site + 1, max(2, n))  # include self
    try:
        dists, idxs = neighbor_index.tree.query(centers, k=k)
    except Exception:
        D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        idxs = np.argsort(D, axis=1)[:, :k]

    mmax = 1
    tree = neighbor_index.tree

    for i in range(n):
        ci = centers[i]; ri = radii[i]
        neigh = [int(j) for j in idxs[i] if j != i]
        for j in neigh:
            cj = centers[j]; rj = radii[j]
            v = cj - ci
            d = np.linalg.norm(v)
            if d < 1e-12:
                continue
            # Two spheres intersect if |ri - rj| < d < ri + rj
            if not (abs(ri - rj) < d < (ri + rj)):
                continue

            # Circle center along chord and radius
            t = (ri*ri - rj*rj + d*d) / (2*d*d)
            center_circle = ci + t * v
            h2 = ri*ri - np.dot(center_circle - ci, center_circle - ci)
            if h2 <= 0:
                h = 0.0
            else:
                h = np.sqrt(h2)

            # Build orthonormal basis {u, w} perpendicular to axis (v_hat)
            v_hat = v / d
            if abs(v_hat[0]) < 0.9:
                tmp = np.array([1.0, 0.0, 0.0])
            else:
                tmp = np.array([0.0, 1.0, 0.0])
            u = np.cross(v_hat, tmp); un = np.linalg.norm(u)
            if un < 1e-12:
                continue
            u /= un
            w = np.cross(v_hat, u)

            # Sample points on the circle
            angles = [0.0] if h < 1e-12 else (np.linspace(0, 2*np.pi, samples_per_circle, endpoint=False))
            for ang in angles:
                p = center_circle if h < 1e-12 else (center_circle + h*np.cos(ang)*u + h*np.sin(ang)*w)

                # Query neighbors within max possible radius
                maxR = np.max(radii)
                q_idx = tree.query_ball_point(p, maxR + 2*tol_abs)
                if not q_idx:
                    continue

                dists_pc = np.linalg.norm(centers[q_idx] - p, axis=1)
                r_q = radii[q_idx]
                err = np.abs(dists_pc - r_q)
                thr = tol_abs + tol_rel * np.maximum(1.0, r_q)
                m = int(np.sum(err <= thr))
                if m > mmax:
                    mmax = m
                    if early_stop_N is not None and mmax >= early_stop_N:
                        return mmax
    return mmax


# --------------------------------------------------------
# Public: minimal s for N-way surface intersection (fixed ratios)
# --------------------------------------------------------
def s_star_fixed_ratios(centers, alpha_idx, neighbor_index, N_max=6,
                        eps=1e-8, kNN=32,
                        max_postpass_scales=12,   # cap number of scales to validate
                        samples_per_circle=8,     # fast validator sampling
                        max_pairs_per_site=48):   # bound pair enumeration per site
    """
    Returns dict {N: s_N_star} for N=1..N_max (np.inf if not found in window).
    Pipeline:
      1) k-NN pairs → exact s2
      2) k-NN triples → exact s3
      3) k-NN 4-tuples → Apollonius solve → (s, x) and local multiplicity count
      4) FAST post-pass at a bounded set of candidate s using pair-circle sampling
    """
    s_star = {N: np.inf for N in range(1, N_max + 1)}
    s_star[1] = 0.0

    n = len(centers)
    if n <= 1:
        return s_star

    # Characteristic spacing for local multiplicity search radius
    diffs = centers[1:] - centers[0]
    a_char = float(np.min(np.linalg.norm(diffs, axis=1))) if len(diffs) else 1.0
    local_r = 3.0 * a_char  # used in quad-counting (not in fast validator)

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

            # Pairwise LB on s for this quadruple
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

    # -------- Post-pass multiplicity checks (FAST) --------
    # Build candidate s set: {s2, s3} U smallest M from quads, dedup within rel tol.
    cand = []
    if np.isfinite(s2): cand.append(float(s2))
    if np.isfinite(s3): cand.append(float(s3))
    quad_s_candidates.sort()
    for s in quad_s_candidates[:max_postpass_scales]:
        cand.append(float(s))

    # Deduplicate (relative tol 1e-8)
    cand_sorted = []
    for s in sorted(cand):
        if not cand_sorted or abs(s - cand_sorted[-1]) > 1e-8 * max(1.0, cand_sorted[-1]):
            cand_sorted.append(s)

    # Validate each s via pair-circle sampler; early-stop per s when reaching N_max
    for s in cand_sorted:
        mmax = _fast_visual_validator_at_s(
            centers, alpha_idx, neighbor_index, s, a_char,
            samples_per_circle=samples_per_circle,
            tol_abs=1e-3, tol_rel=1e-6,
            max_pairs_per_site=max_pairs_per_site,
            early_stop_N=N_max
        )
        if mmax >= 1:
            for N in range(1, min(N_max, mmax) + 1):
                if s < s_star[N]:
                    s_star[N] = s

    return s_star
