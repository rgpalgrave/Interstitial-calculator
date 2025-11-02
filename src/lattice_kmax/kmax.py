import numpy as np
from itertools import combinations

def _three_sphere_intersections(c1,r1,c2,r2,c3,r3,eps=1e-12):
    # Reduce to line-sphere intersection (sphere 1 at origin frame)
    e1, e2 = c2-c1, c3-c1
    A = np.vstack([2*e1, 2*e2])         # plane system (2x3)
    b = np.array([r1**2 - r2**2 + np.dot(c2,c2) - np.dot(c1,c1),
                  r1**2 - r3**2 + np.dot(c3,c3) - np.dot(c1,c1)])
    U,S,Vt = np.linalg.svd(A, full_matrices=True)
    if (S > 1e-12).sum() < 2: return []
    # Reconstruct: pad S with zeros to match dimensions
    S_inv = np.zeros(3)
    S_inv[:len(S)] = 1.0 / S
    x0 = Vt.T @ np.diag(S_inv) @ U.T @ b  # one point on the line
    v  = Vt.T[:, -1]  # null space vector
    a = np.dot(v,v)
    bq = 2*np.dot(x0, v)
    cq = np.dot(x0,x0) - r1*r1
    disc = bq*bq - 4*a*cq
    if disc < -eps: return []
    if abs(disc) <= eps:
        t = -bq/(2*a)
        return [c1 + x0 + t*v]
    rt = np.sqrt(max(0.0, disc))
    t1, t2 = (-bq-rt)/(2*a), (-bq+rt)/(2*a)
    return [c1 + x0 + t1*v, c1 + x0 + t2*v]

def kmax_surface(centers, radii, neighbor_index, eps=1e-8, local_radius=None):
    kmax = 0
    if local_radius is None:
        # heuristic: enough to catch same-shell neighbors
        diffs = centers[1:] - centers[0]
        a = np.min(np.linalg.norm(diffs, axis=1))
        local_radius = 2.5*a
    for i in range(len(centers)):
        nbr = [j for j in neighbor_index.list_for(i) if j>i]
        for j,k in combinations(nbr, 2):
            for x in _three_sphere_intersections(centers[i],radii[i],
                                                 centers[j],radii[j],
                                                 centers[k],radii[k]):
                loc = neighbor_index.local_around_point(x, local_radius)
                d = np.linalg.norm(centers[loc]-x, axis=1)
                m = int(np.sum(np.abs(d - radii[loc]) <= eps))
                if m > kmax: kmax = m
    return kmax
