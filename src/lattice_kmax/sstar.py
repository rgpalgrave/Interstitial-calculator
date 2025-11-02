import numpy as np
from itertools import combinations
from .utils import pairwise_lower_bound_s

def _quadruple_s_and_center(centers, idxs, alpha_idx, eps=1e-16):
    # Solve for x(s^2) and tangency on first sphere → quadratic in lam=s^2
    P = np.vstack([centers[i] for i in idxs])    # 4x3
    al = np.array([alpha_idx[i] for i in idxs])  # 4
    c0 = P[0]
    A = 2.0*(P[1:] - c0)                         # 3x3
    if np.linalg.matrix_rank(A) < 3: return None
    b0 = np.sum(P[1:]**2, axis=1) - np.sum(c0**2)
    d  = (al[1:]**2 - al[0]**2)
    Ainv = np.linalg.inv(A)
    x0 = Ainv @ b0
    x1 = -Ainv @ d

    # Tangency on sphere 0: |x|^2 - 2c0·x + |c0|^2 - lam*al0^2 = 0
    a = np.dot(x1,x1)
    b = 2*np.dot(x0,x1) - 2*np.dot(c0,x1) - (al[0]**2)
    c = np.dot(x0,x0) - 2*np.dot(c0,x0) + np.dot(c0,c0)

    if abs(a) < eps:
        if abs(b) < eps: return None
        lam = -c/b
        if lam < 0: return None
    else:
        disc = b*b - 4*a*c
        if disc < 0: return None
        r1 = (-b - np.sqrt(disc)) / (2*a)
        r2 = (-b + np.sqrt(disc)) / (2*a)
        candidates = [r for r in (r1,r2) if r >= 0]
        if not candidates: return None
        lam = min(candidates)
    s = np.sqrt(lam)
    x = x0 + lam*x1
    return s, x

def s_star_fixed_ratios(centers, alpha_idx, neighbor_index, N_max=6, eps=1e-8):
    s_star = {N: np.inf for N in range(1, N_max+1)}
    # precompute a small local search radius
    diffs = centers[1:] - centers[0]
    a = np.min(np.linalg.norm(diffs, axis=1))
    local_r = 2.5*a

    n = len(centers)
    for i in range(n):
        nbr = [j for j in neighbor_index.list_for(i) if j>i]
        # enumerate local quadruples containing i
        for j,k,l in combinations(nbr, 3):
            idxs = (i,j,k,l)
            # lower bound prune
            al = np.array([alpha_idx[t] for t in idxs])
            lb = pairwise_lower_bound_s(centers, idxs, al)
            if lb >= s_star.get(N_max, np.inf):  # if already worse than best 6-fold
                continue
            out = _quadruple_s_and_center(centers, idxs, alpha_idx)
            if out is None: continue
            s, x = out
            if s >= s_star.get(N_max, np.inf):  # prune again
                continue
            # multiplicity at (x,s)
            loc = neighbor_index.local_around_point(x, local_r)
            d = np.linalg.norm(centers[loc]-x, axis=1)
            targ = s * np.asarray([alpha_idx[t] for t in loc])
            m = int(np.sum(np.abs(d - targ) <= eps))
            if m >= 1:
                for N in range(1, min(N_max, m)+1):
                    if s < s_star[N]: s_star[N] = s
    return s_star
