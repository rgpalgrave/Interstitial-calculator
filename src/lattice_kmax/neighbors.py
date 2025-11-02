import numpy as np
from scipy.spatial import cKDTree

class NeighborIndex:
    def __init__(self, centers: np.ndarray, rcut: float):
        self.centers = centers
        self.tree = cKDTree(centers)
        self.rcut = rcut

    def list_for(self, i: int):
        return self.tree.query_ball_point(self.centers[i], self.rcut)

    def local_around_point(self, x: np.ndarray, r: float):
        return self.tree.query_ball_point(x, r)
