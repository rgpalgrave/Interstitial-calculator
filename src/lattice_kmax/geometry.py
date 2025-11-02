import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class LatticeSpec:
    kind: str                    # 'sc','bcc','fcc','hcp','generic'
    a: float
    b: float = None              # optional (orthorhombic/tetragonal/hex)
    c: float = None
    angles: Tuple[float,float,float] = None  # alpha,beta,gamma in degrees (generic)
    offset: Tuple[float,float,float] = (0.0,0.0,0.0)  # fractional offset
    supercell: Tuple[int,int,int] = (5,5,5)

def _basis_matrix(spec: LatticeSpec) -> np.ndarray:
    k = spec.kind.lower()
    a = spec.a
    if k == "sc":
        return np.diag([a,a,a])
    if k == "bcc":  # conventional cubic with two-point basis handled later
        return np.diag([a,a,a])
    if k == "fcc":
        return np.diag([a,a,a])
    if k == "hcp":
        a, c = spec.a, (spec.c if spec.c else np.sqrt(8/3)*spec.a)
        return np.array([[a,0,0],
                         [a/2, np.sqrt(3)*a/2, 0],
                         [0,0,c]])
    if k == "generic":
        # Full Bravais from (a,b,c,alpha,beta,gamma)
        a,b,c = spec.a, (spec.b or spec.a), (spec.c or spec.a)
        alpha,beta,gamma = (ang*np.pi/180 for ang in (spec.angles or (90,90,90)))
        v_x = np.array([a, 0, 0])
        v_y = np.array([b*np.cos(gamma), b*np.sin(gamma), 0])
        cx = c*np.cos(beta)
        cy = c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)
        cz = np.sqrt(c**2 - cx**2 - cy**2)
        v_z = np.array([cx, cy, cz])
        return np.vstack([v_x, v_y, v_z]).T
    raise ValueError(f"Unknown lattice kind: {spec.kind}")

def _basis_points(kind: str) -> np.ndarray:
    k = kind.lower()
    if k == "sc":  return np.array([[0,0,0]])
    if k == "bcc": return np.array([[0,0,0],[0.5,0.5,0.5]])
    if k == "fcc": return np.array([[0,0,0],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5]])
    if k == "hcp": return np.array([[0,0,0],[2/3,1/3,1/2]])
    if k == "generic": return np.array([[0,0,0]])
    raise ValueError

def generate_points(specs: List[LatticeSpec]) -> tuple[np.ndarray, np.ndarray]:
    """Return (centers, lattice_index_per_point)."""
    all_pts, labels = [], []
    for li, spec in enumerate(specs):
        B = _basis_matrix(spec)
        basis = _basis_points(spec.kind)
        nx,ny,nz = spec.supercell
        off = np.asarray(spec.offset)
        for i in range(-nx, nx+1):
            for j in range(-ny, ny+1):
                for k in range(-nz, nz+1):
                    t = np.array([i,j,k], dtype=float)
                    cell_origin = B @ t
                    for b in basis:
                        frac = (b + off) % 1.0
                        pt = cell_origin + B @ frac
                        all_pts.append(pt)
                        labels.append(li)
    return np.array(all_pts, dtype=float), np.array(labels, dtype=int)
