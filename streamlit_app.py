# streamlit_app.py
# Single-page Streamlit UI for k_max and s_N* with support for 1–2 lattices

import os, sys
# If you kept a src/ layout, ensure imports work on Streamlit Cloud:
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import numpy as np
import pandas as pd

from lattice_kmax.geometry import LatticeSpec
from lattice_kmax.cache import prepare_geometry
from lattice_kmax.kmax import kmax_surface
from lattice_kmax.sstar import s_star_fixed_ratios
from lattice_kmax.fastscan import blazing_s_star

# Optional: show package version to ensure hot reloads after edits
try:
    import lattice_kmax as lk
    PKG_VER = getattr(lk, "__version__", "dev")
except Exception:
    PKG_VER = "dev"

st.set_page_config(page_title="Lattice k_max / s_N* Explorer", layout="wide")
st.title("Lattice k_max / s_N* Explorer")
st.caption(f"package version: {PKG_VER}")

# ---------- Sidebar: global numeric settings ----------
with st.sidebar:
    st.header("Numerics")
    eps = st.number_input("Surface tolerance ε (units of a)", value=1e-8, format="%.1e")
    rcut_factor = st.number_input("Neighbour cutoff factor ×a", value=2.2, step=0.1, format="%.2f")
    kNN = st.slider("k-NN (pairs/triangles/tetra)", min_value=8, max_value=64, value=32, step=2)
    nmax = st.slider("Max N for s_N*", 1, 12, 6)
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Computation mode", ["Exact (quad solve)", "Blazing (visual-style)"], index=0)


# ---------- Helper to build one lattice block ----------
def lattice_block(name: str, default_kind="fcc", default_a=1.0, default_offset=(0.0,0.0,0.0)):
    st.subheader(name)
    col1, col2 = st.columns(2)
    with col1:
        kind = st.selectbox(f"{name} — lattice type", ["sc","bcc","fcc","hcp","generic"], index=["sc","bcc","fcc","hcp","generic"].index(default_kind), key=f"{name}_kind")
        a = st.number_input(f"{name} — a", value=float(default_a), min_value=1e-6, step=0.1, format="%.6f", key=f"{name}_a")
    with col2:
        if kind in ("hcp","generic"):
            c_default = np.sqrt(8/3)*a if kind=="hcp" else a
            c = st.number_input(f"{name} — c", value=float(c_default), step=0.1, format="%.6f", key=f"{name}_c")
        else:
            c = None

    b = None; angles = None
    if kind == "generic":
        st.caption(f"{name}: generic cell (supply b and angles in °)")
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            b = st.number_input(f"{name} — b", value=float(a), step=0.1, format="%.6f", key=f"{name}_b")
        with col4:
            alpha = st.number_input(f"{name} — α (°)", value=90.0, step=1.0, format="%.3f", key=f"{name}_alpha")
        with col5:
            beta  = st.number_input(f"{name} — β (°)", value=90.0, step=1.0, format="%.3f", key=f"{name}_beta")
        with col6:
            gamma = st.number_input(f"{name} — γ (°)", value=90.0, step=1.0, format="%.3f", key=f"{name}_gamma")
        angles = (alpha, beta, gamma)

    st.caption(f"{name}: fractional offsets")
    ox, oy, oz = st.columns(3)
    with ox:
        off_x = st.number_input(f"{name} — off x", value=float(default_offset[0]), step=0.05, format="%.6f", key=f"{name}_offx")
    with oy:
        off_y = st.number_input(f"{name} — off y", value=float(default_offset[1]), step=0.05, format="%.6f", key=f"{name}_offy")
    with oz:
        off_z = st.number_input(f"{name} — off z", value=float(default_offset[2]), step=0.05, format="%.6f", key=f"{name}_offz")

    return dict(kind=kind, a=a, b=b, c=c, angles=angles, offset=(off_x, off_y, off_z))

# ---------- Geometry: supercell window (shared for both lattices) ----------
st.sidebar.header("Supercell window")
nx = st.sidebar.slider("± cells in x", 2, 8, 5)
ny = st.sidebar.slider("± cells in y", 2, 8, 5)
nz = st.sidebar.slider("± cells in z", 2, 8, 5)
supercell = (nx, ny, nz)

st.sidebar.header("Radii (fixed ratios)")
st.sidebar.caption("Provide α per lattice, in the order they are defined below.")
alpha_text = st.sidebar.text_input("α ratios (comma separated)", value="1.0")
alphas_user_raw = [x.strip() for x in alpha_text.split(",") if x.strip()]

# ---------- Lattice 1 ----------
st.markdown("### Lattice 1")
lat1 = lattice_block("Lattice1", default_kind="fcc", default_a=1.0, default_offset=(0,0,0))

# ---------- Optional Lattice 2 ----------
st.markdown("---")
use_second = st.checkbox("Add Lattice 2", value=False)
if use_second:
    st.markdown("### Lattice 2")
    lat2 = lattice_block("Lattice2", default_kind="bcc", default_a=1.0, default_offset=(0.5,0.5,0.5))
else:
    lat2 = None

# Ensure we have correct number of alphas
num_lattices = 2 if use_second else 1
if len(alphas_user_raw) < num_lattices:
    # pad with last value or 1.0
    pad_val = float(alphas_user_raw[-1]) if alphas_user_raw else 1.0
    alphas_user = [float(x) for x in alphas_user_raw] + [pad_val]*(num_lattices - len(alphas_user_raw))
else:
    alphas_user = [float(x) for x in alphas_user_raw[:num_lattices]]

# ---------- Build LatticeSpec list ----------
specs = []
specs.append(LatticeSpec(
    kind=lat1["kind"], a=lat1["a"], b=lat1["b"], c=lat1["c"], angles=lat1["angles"],
    offset=lat1["offset"], supercell=supercell
))
if use_second:
    specs.append(LatticeSpec(
        kind=lat2["kind"], a=lat2["a"], b=lat2["b"], c=lat2["c"], angles=lat2["angles"],
        offset=lat2["offset"], supercell=supercell
    ))

# ---------- Cache geometry build ----------
@st.cache_resource(show_spinner=False)
def build_geometry(specs_tuple, alphas_tuple, rcut_factor):
    # specs_tuple: tuple of serialisable dicts
    specs_objs = [LatticeSpec(**d) for d in specs_tuple]
    geom = prepare_geometry(specs_objs, list(alphas_tuple), rcut_factor=rcut_factor)
    return geom

# Convert specs to serialisable tuples for the cache key
specs_for_cache = tuple(dict(
    kind=s.kind, a=float(s.a),
    b=None if s.b is None else float(s.b),
    c=None if s.c is None else float(s.c),
    angles=None if s.angles is None else tuple(float(x) for x in s.angles),
    offset=tuple(float(x) for x in s.offset),
    supercell=tuple(int(x) for x in s.supercell)
) for s in specs)

with st.spinner("Preparing geometry..."):
    geom = build_geometry(specs_for_cache, tuple(float(x) for x in alphas_user), rcut_factor)

st.success(f"Points: {len(geom.centers)} | lattices: {num_lattices} | k-NN: {kNN} | r_cut≈{rcut_factor}·a")

# ---------- k_max at fixed scale ----------
st.markdown("---")
st.header("k_max at fixed scale")
s_scale = st.number_input("Scale s (r_i = s * α_lattice(i))", value=0.60, min_value=0.0, step=0.05, format="%.6f")
run_kmax = st.button("Compute k_max", type="primary")

if run_kmax:
    radii = s_scale * geom.alpha_idx  # per-center radii, mapped from lattice idx
    with st.spinner("Computing k_max (surface intersections)..."):
        k_val = kmax_surface(geom.centers, radii, geom.neighbors, eps=eps)
    st.subheader("k_max result")
    st.write(f"**k_max = {k_val}** at scale **s = {s_scale:.6f}**")

# ---------- s_N* (minimal scales) ----------
st.markdown("---")
st.header("Minimal scales s_N* (fixed ratios)")
run_sstar = st.button(f"Compute s_N* for N = 1..{nmax}", type="secondary")

if run_sstar:
    with st.spinner(f"Computing s_N* (kNN={kNN}, ε={eps:g})..."):
        sstar = s_star_fixed_ratios(
            geom.centers, geom.alpha_idx, geom.neighbors,
            N_max=nmax, eps=eps, kNN=kNN
        )
    st.subheader("Minimal scales s_N*")
    data = [{"N": N, "s_N*": (np.inf if np.isinf(s) else s)} for N, s in sstar.items()]
    df = pd.DataFrame(data).sort_values("N")
    st.dataframe(df, use_container_width=True)
    plot_df = df.replace({np.inf: np.nan}).set_index("N")
    st.bar_chart(plot_df)
    st.caption("NaN = not found within current window/tolerance.")

st.info("Tips: increase supercell and/or k-NN if results look window-dependent; tighten ε to confirm exact symmetries.")

st.header("Minimal scales s_N* (fixed ratios)")
run_sstar = st.button(f"Compute s_N* for N = 1..{nmax}", type="secondary")

if run_sstar:
    if mode.startswith("Exact"):
        with st.spinner(f"Computing s_N* (kNN={kNN}, ε={eps:g})..."):
            sstar = s_star_fixed_ratios(
                geom.centers, geom.alpha_idx, geom.neighbors,
                N_max=nmax, eps=eps, kNN=kNN
            )
    else:
        with st.spinner("Computing s_N* (Blazing mode: pair-circle sampling)..."):
            sstar = blazing_s_star(
                geom.centers, geom.alpha_idx, geom.neighbors,
                N_max=nmax,
                k_pairs=48,            # speed knob
                samples_per_circle=8,  # 6–8 typical
                tol_abs=1e-3,          # 1–2e-3 good with a=1 scaling
                tol_rel=1e-6,
                max_events=2048
            )
