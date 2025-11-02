import json, sys
import numpy as np
import typer
from .geometry import LatticeSpec
from .cache import prepare_geometry
from .kmax import kmax_surface
from .sstar import s_star_fixed_ratios

app = typer.Typer(add_completion=False)

@app.command("kmax")
def kmax_cmd(
    lattice: str = typer.Option("fcc"),
    a: float = typer.Option(1.0),
    offset: str = typer.Option("0,0,0"),   # fractional
    supercell: str = typer.Option("5,5,5"),
    s: float = typer.Option(0.6),
    alphas: str = typer.Option("1.0"),     # one per lattice in order of specs
    eps: float = typer.Option(1e-8)
):
    off = tuple(float(x) for x in offset.split(","))
    sc = tuple(int(x) for x in supercell.split(","))
    specs = [LatticeSpec(kind=lattice, a=a, offset=off, supercell=sc)]
    alpha_list = [float(x) for x in alphas.split(",")]
    geom = prepare_geometry(specs, alpha_list)
    radii = s * geom.alpha_idx
    kmax = kmax_surface(geom.centers, radii, geom.neighbors, eps=eps)
    print(kmax)

@app.command("sstar")
def sstar_cmd(
    lattice: str = typer.Option("fcc"),
    a: float = typer.Option(1.0),
    offset: str = typer.Option("0,0,0"),
    supercell: str = typer.Option("5,5,5"),
    alphas: str = typer.Option("1.0"),
    nmax: int = typer.Option(6),
    eps: float = typer.Option(1e-8),
):
    off = tuple(float(x) for x in offset.split(","))
    sc = tuple(int(x) for x in supercell.split(","))
    specs = [LatticeSpec(kind=lattice, a=a, offset=off, supercell=sc)]
    alpha_list = [float(x) for x in alphas.split(",")]
    geom = prepare_geometry(specs, alpha_list)
    out = s_star_fixed_ratios(geom.centers, geom.alpha_idx, geom.neighbors, N_max=nmax, eps=eps)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    app()
