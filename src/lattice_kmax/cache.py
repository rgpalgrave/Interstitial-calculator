from dataclasses import dataclass
import numpy as np
from .geometry import LatticeSpec, generate_points
from .neighbors import NeighborIndex

@dataclass
class GeometryCache:
    centers: np.ndarray
    lattice_idx: np.ndarray
    alpha_idx: np.ndarray
    neighbors: NeighborIndex

def prepare_geometry(specs, alphas, rcut_factor=2.2):
    centers, lattice_idx = generate_points(specs)
    # characteristic spacing ~ shortest non-zero vector from origin to others
    diffs = centers[1:] - centers[0]
    a = np.min(np.linalg.norm(diffs, axis=1))
    neighbors = NeighborIndex(centers, rcut_factor*a)
    alpha_idx = np.array([alphas[i] for i in lattice_idx], dtype=float)
    return GeometryCache(centers, lattice_idx, alpha_idx, neighbors)
