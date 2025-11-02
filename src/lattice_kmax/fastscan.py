# src/lattice_kmax/fastscan.py
# v0.1.0 — Event-driven "blazing" s_N* using pair-circle sampling + bisection
import numpy as np
from itertools import combinations

# ------------------------------
# Core: pair-circle multiplicity
# ------------------------------
def _pair_circle_samples(ci, ri, cj, rj, samples_per_circle):
    """
    Return sample points on the intersection circle of two spheres (if any).
    """
    v = cj - ci
    d = np.linalg.norm(v)
    if d < 1e-12:
        return []  # coincident centers; ignore
    # Intersect condition
    if not (abs(ri - rj) < d < (ri + rj)):
        # Tangency: still one point exactly; treat as tiny circle with 1 sample
        if abs(d - (ri + rj)) < 1e-12 or abs(d - abs(ri - rj)) < 1e-12:
            t = (ri*ri - rj*rj + d*d) / (2*d*d)
            center_circle = ci + t * v
            return [center_circle]
        return []

    t = (ri*ri - rj*rj + d*d) / (2*d*d)
    center_circle = ci + t * v
    h2 = ri*ri - np.dot(center_circle - ci, center_circle - ci)
    if h2 <= 0:
        return [center_circle]
    h = np.sqrt(h2)

    v_hat = v / d
    # pick any vector not parallel to v_hat
    if abs(v_hat[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])
    u = np.cross(v_hat, tmp); un = np.linalg.norm(u)
    if un < 1e-12:
        return []
    u /= un
    w = np.cross(v_hat, u)

    if samples_per_circle <= 1:
        return [center_circle]
    ang = np.linspace(0, 2*np.pi, samples_per_circle, endpoint=False)
    pts = center_circle + h*np.cos(ang)[:, None]*u + h*np.sin(ang)[:, None]*w
    return [p for p in pts]


def _multiplicity_at_points(points, centers, radii, kdtree, tol_abs, tol_rel):
    """
    Count maximum number of spheres whose surfaces pass through each sample point.
    """
    if len(points) == 0:
        return 1
    mmax = 1
    maxR = float(np.max(radii)) if len(radii) else 0.0
    for p in points:
        # query neighbors possibly relevant for equality |p-c| ~ r
        idxs = kdtree.query_ball_point(p, maxR + 2*tol_abs)
        if not idxs:
            continue
        dists = np.linalg.norm(centers[idxs] - p, axis=1)
        r_loc = radii[idxs]
        err = np.abs(dists - r_loc)
        thr = tol_abs + tol_rel * np.maximum(1.0, r_loc)
        m = int(np.sum(err <= thr))
        if m > mmax:
            mmax = m
    return mmax


def _estimate_multiplicity_at_s(centers, alpha_idx, neighbor_index, s,
                                k_pairs=48, samples_per_circle=8,
                                tol_abs=1e-3, tol_rel=1e-6):
    """
    Approximate m(s) (max surface-intersection multiplicity) at scale s
    using pair-circle sampling around each site with a bounded neighbor set.
    """
    if not np.isfinite(s) or s <= 0:
        return 1
    radii = s * alpha_idx
    n = len(centers)
    if n == 0:
        return 1

    k = min(max(2, k_pairs + 1), max(2, n))  # include self
    try:
        dists, idxs = neighbor_index.tree.query(centers, k=k)
    except Exception:
        D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        idxs = np.argsort(D, axis=1)[:, :k]

    mmax = 1
    kd = neighbor_index.tree
    for i in range(n):
        ci = centers[i]; ri = radii[i]
        neigh = [int(j) for j in idxs[i] if j != i]
        for j in neigh:
            cj = centers[j]; rj = radii[j]
            pts = _pair_circle_samples(ci, ri, cj, rj, samples_per_circle)
            if not pts:
                continue
            m = _multiplicity_at_points(pts, centers, radii, kd, tol_abs, tol_rel)
            if m > mmax:
                mmax = m
                # early best possible guard (not strictly bounded, but good practice)
                if mmax >= 32:
                    return mmax
    return mmax


# ------------------------------
# Candidate s from local geometry
# ------------------------------
def candidate_scales_from_pairs(centers, alpha_idx, neighbor_index,
                                k_pairs=48, dedup_rtol=1e-8):
    """
    Event list: s_pair = d / (ai + aj) for bounded neighbor pairs.
    """
    n = len(centers)
    if n == 0:
        return np.array([], dtype=float)
    k = min(max(2, k_pairs + 1), max(2, n))
    try:
        dists, idxs = neighbor_index.tree.query(centers, k=k)
    except Exception:
        D = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        idxs = np.argsort(D, axis=1)[:, :k]

    cand = []
    for i in range(n):
        ai = alpha_idx[i]
        for j in idxs[i][1:]:
            if j <= i:
                continue
            aj = alpha_idx[int(j)]
            d = np.linalg.norm(centers[i] - centers[int(j)])
            denom = ai + aj
            if denom <= 0:
                continue
            s = d / denom
            cand.append(s)

    if not cand:
        return np.array([], dtype=float)
    cand.sort()
    # deduplicate within relative tolerance
    uniq = [cand[0]]
    for x in cand[1:]:
        if abs(x - uniq[-1]) > dedup_rtol * max(1.0, uniq[-1]):
            uniq.append(x)
    return np.asarray(uniq, dtype=float)


# ---------------
# Bisection refine
# ---------------
def _refine_threshold_s(centers, alpha_idx, neighbor_index, target_N,
                        s_left, s_right,
                        validator_kwargs,
                        max_iter=24):
    """
    Given m(s_left) < target_N and m(s_right) >= target_N (or equal), refine the minimal s.
    """
    m_left = _estimate_multiplicity_at_s(centers, alpha_idx, neighbor_index, s_left, **validator_kwargs)
    m_right = _estimate_multiplicity_at_s(centers, alpha_idx, neighbor_index, s_right, **validator_kwargs)
    if m_right < target_N:
        return np.inf  # not actually crossed here
    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (s_left + s_right)
        m_mid = _estimate_multiplicity_at_s(centers, alpha_idx, neighbor_index, mid, **validator_kwargs)
        if m_mid >= target_N:
            s_right = mid
            m_right = m_mid
        else:
            s_left = mid
            m_left = m_mid
        # stop when close enough
        if abs(s_right - s_left) <= 1e-8 * max(1.0, s_right):
            break
    return s_right


# ------------------------------
# Public: blazing s_N* estimator
# ------------------------------
def blazing_s_star(centers, alpha_idx, neighbor_index,
                   N_max=6,
                   k_pairs=48,
                   samples_per_circle=8,
                   tol_abs=1e-3,
                   tol_rel=1e-6,
                   max_events=2048):
    """
    Fast, approximate (but robust in practice) minimal scales s_N* for N=1..N_max.

    Strategy:
      1) Build event list S = { d_ij / (a_i + a_j) } from bounded neighbor pairs.
      2) Sweep S in ascending order. At each candidate s:
         - Evaluate m(s) with pair-circle sampling.
         - When m(s) reaches a new N, refine with bisection on s across the last drop.
      3) Stop early if we have s_N* for all N <= N_max.

    Returns: dict {N: s_N*} (np.inf if not reached within scanned events).
    """
    out = {N: np.inf for N in range(1, N_max + 1)}
    out[1] = 0.0

    events = candidate_scales_from_pairs(centers, alpha_idx, neighbor_index,
                                         k_pairs=k_pairs, dedup_rtol=1e-8)
    if len(events) == 0:
        return out

    # cap events for speed
    if len(events) > max_events:
        events = events[:max_events]

    validator_kwargs = dict(
        k_pairs=k_pairs,
        samples_per_circle=samples_per_circle,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
    )

    # sweep
    prev_s = 0.0
    prev_m = _estimate_multiplicity_at_s(centers, alpha_idx, neighbor_index, prev_s, **validator_kwargs)
    # track which N are still open
    open_N = set(range(2, N_max + 1))

    for s in events:
        m_now = _estimate_multiplicity_at_s(centers, alpha_idx, neighbor_index, s, **validator_kwargs)

        # for each N we haven't solved yet, if prev_m < N <= m_now, refine threshold
        newly_satisfied = []
        for N in sorted(list(open_N)):
            if prev_m < N <= m_now:
                sN = _refine_threshold_s(centers, alpha_idx, neighbor_index, N, prev_s, s,
                                         validator_kwargs, max_iter=24)
                out[N] = min(out[N], sN)
                newly_satisfied.append(N)

        for N in newly_satisfied:
            open_N.discard(N)

        if not open_N:
            break

        prev_s, prev_m = s, m_now

    return out
