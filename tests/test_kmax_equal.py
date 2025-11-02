import numpy as np
from lattice_kmax.geometry import LatticeSpec
from lattice_kmax.cache import prepare_geometry
from lattice_kmax.kmax import kmax_surface

def test_kmax_fcc_equal():
    specs = [LatticeSpec(kind="fcc", a=1.0, offset=(0,0,0), supercell=(3,3,3))]
    geom = prepare_geometry(specs, [1.0])
    # pick a radius where 4-way at tetra centers is expected before 6-way
    s = 0.612  # heuristic; this is a smoke test (not exact)
    radii = s * geom.alpha_idx
    k = kmax_surface(geom.centers, radii, geom.neighbors, eps=1e-6)
    assert k >= 4
