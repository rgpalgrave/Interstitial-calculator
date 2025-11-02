import numpy as np

def pairwise_lower_bound_s(centers, idxs, alphas):
    # s >= max_{pairs} |ci-cj| / (alpha_i + alpha_j)
    max_lb = 0.0
    for a in range(4):
        for b in range(a+1,4):
            i,j = idxs[a], idxs[b]
            d = np.linalg.norm(centers[i]-centers[j])
            lb = d / (alphas[a] + alphas[b])
            if lb > max_lb: max_lb = lb
    return max_lb
