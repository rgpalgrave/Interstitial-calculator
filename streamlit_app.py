# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd

from lattice_kmax.geometry import LatticeSpec
from lattice_kmax.cache import prepare_geometry
from lattice_kmax.neighbors import NeighborIndex
from lattice_kmax.kmax import kmax_surface
from lattice_kmax.sstar import s_star_fixed_ratios

st.set_page_config(page_title="Lattice k_max / s_N* Explorer", layout="wide")
st.title("Lattice k_max / s_N* Explorer")

with st.sidebar:
    st.header("Geometry")
    lattice = st.selectbox("Lattice type", ["sc","bcc","fcc","hcp","generic"], index=2)
    a = st.number_input("a (Å or unit length)", value=1.0, min_value=1e-6, step=0.1, format="%.6f")

    if lattice in ("hcp","generic"):
        c = st.number_input("c (only for hcp/generic)", value=(np.sqrt(8/3)*a if lattice=="hcp" else a), step=0.1, format="%.6f")
    else:
        c = None

    if lattice == "generic":
        st.caption("Generic cell: supply b and angles (deg)")
        b = st.number_input("b", value=a, step=0.1, format="%.6f")
        alpha = st.number_input("alpha (°)", value=90.0, step=1.0, format="%.3f")
        beta  = st.number_input("beta (°)",  value=90.0, step=1.0, format="%.3f")
        gamma = st.number_input("gamma (°)", value=90.0, step=1.0, format="%.3f")
        angles = (alpha,beta,gamma)
    else:
        b, angles = None, None

    st.divider()
    st.subheader("Offsets (fractional)")
    off_x = st.number_input("offset x", value=0.0, step=0.05, format="%.6f")
    off_y = st.number_input("offset y", value=0.0, step=0.05, format="%.6f")
    off_z = st.number_input("offset z", value=0.0, step=0.05, format="%.6f")
    offset = (off_x, off_y, off_z)

    st.divider()
    st.subheader("Supercell window")
    nx = st.slider("± cells in x", 2, 8, 5)
    ny = st.slider("± cells in y", 2, 8, 5)
    nz = st.slider("± cells in z", 2, 8, 5)
    supercell = (nx, ny, nz)

    st.divider()
    st.header("Radii (fixed ratios)")
    alpha_text = st.text_input("α ratios (comma-sep, per lattice spec order)", value="1.0")
    alphas_user = [float(x.strip()) for x in alpha_text.split(",") if x.strip()]

    st.caption("Most use-cases = one lattice → provide a single α value (e.g. 1.0).")

    st.divider()
    st.header("Numerics")
    eps = st.number_input("Surface tolerance ε (in units of a)", value=1e-8, format="%.1e")
    rcut_factor = st.number_input("Neighbour cutoff factor ×a", value=2.2, step=0.1, format="%.2f")
    nmax = st.slider("Max N for s_N*", 1, 10, 6)

    st.divider()
    st.header("k_max at fixed scale")
    s_scale = st.number_input("Scale s (r_i = s * α_i)", value=0.60, min_value=0.0, step=0.05, format="%.6f")
    run_kmax = st.button("Compute k_max", type="primary")

    st.divider()
    run_sstar = st.button("Compute s_N* (minimal scales)", type="secondary")

# --- Build lattice specs (single lattice by default; extend if you want multiple) ---
spec = LatticeSpec(
    kind=lattice, a=a, b=b, c=c, angles=angles,
    offset=offset, supercell=supercell
)

@st.cache_resource(show_spinner=False)
def build_geometry(spec: LatticeSpec, alphas: list[float], rcut_factor: float):
    geom = prepare_geometry([spec], alphas, rcut_factor=rcut_factor)
    return geom

with st.spinner("Preparing geometry..."):
    geom = build_geometry(spec, alphas_user, rcut_factor)

st.success(f"Points generated: {len(geom.centers)}  |  neighbour r_cut ≈ {rcut_factor}·a")

# --- k_max ---
if run_kmax:
    radii = s_scale * geom.alpha_idx
    with st.spinner("Computing k_max (surface intersections)..."):
        k = kmax_surface(geom.centers, radii, geom.neighbors, eps=eps)
    st.subheader("k_max result")
    st.write(f"**k_max = {k}** at scale **s = {s_scale:.6f}**")

# --- s_N* ---
if run_sstar:
    with st.spinner(f"Computing s_N* for N=1..{nmax} (fixed ratios)..."):
        sstar = s_star_fixed_ratios(geom.centers, geom.alpha_idx, geom.neighbors, N_max=nmax, eps=eps)
    st.subheader("Minimal scales s_N*")
    # Table
    data = [{"N": N, "s_N*": (np.inf if np.isinf(s) else s)} for N, s in sstar.items()]
    df = pd.DataFrame(data).sort_values("N")
    st.dataframe(df, use_container_width=True)
    # Bar chart (hide inf)
    plot_df = df.replace({np.inf: np.nan}).set_index("N")
    st.bar_chart(plot_df)
    st.caption("NaN means no N-way surface intersection found within the current window/tolerance.")

st.info("Tip: enlarge the supercell and/or r_cut if results look window-dependent; tighten ε to confirm exact symmetries.")
