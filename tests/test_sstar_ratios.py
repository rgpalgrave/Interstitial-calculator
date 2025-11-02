import numpy as np
from lattice_kmax.geometry import LatticeSpec
from lattice_kmax.cache import prepare_geometry
from lattice_kmax.sstar import s_star_fixed_ratios

def test_sstar_basic_fcc():
    specs = [LatticeSpec(kind="fcc", a=1.0, supercell=(3,3,3))]
    geom = prepare_geometry(specs, [1.0])
    out = s_star_fixed_ratios(geom.centers, geom.alpha_idx, geom.neighbors, N_max=4, eps=1e-6)
    assert out[1] < np.inf and out[4] < np.inf
