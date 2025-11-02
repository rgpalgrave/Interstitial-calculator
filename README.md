# lattice-kmax

N-way sphere surface intersections on Bravais lattices

## Installation

### Local development

```bash
pip install -e .
```

This installs the package in editable mode, making all imports work properly.

### For Streamlit deployment

Your `requirements.txt` includes `-e .`, which tells the deployment platform to install the package in editable mode. This ensures the imports work on platforms like Streamlit Cloud.

## Running the Streamlit app

```bash
streamlit run streamlit_app.py
```

## Project structure

```
lattice-kmax/
├── pyproject.toml              # Package configuration
├── requirements.txt            # Dependencies (includes -e .)
├── streamlit_app.py            # Main Streamlit web app
├── src/
│   └── lattice_kmax/           # Main package
│       ├── __init__.py         # Package marker (newly added)
│       ├── geometry.py         # Lattice geometry utilities
│       ├── cache.py            # Geometry caching
│       ├── neighbors.py        # Neighbor index for spatial queries
│       ├── kmax.py             # k_max surface computation
│       ├── sstar.py            # s_N* minimal scales solver
│       ├── utils.py            # Utility functions
│       └── cli.py              # Command-line interface
└── tests/
    ├── __init__.py             # Test package marker
    ├── test_kmax_equal.py      # k_max tests
    └── test_sstar_ratios.py    # s_N* tests
```

## Key changes from original

The main fix was adding `src/lattice_kmax/__init__.py` to make the package properly importable. The `requirements.txt` now includes `-e .` which tells pip to install the package from the current directory in editable mode.

When you push this to GitHub:
- Streamlit Cloud will read `requirements.txt`
- See the `-e .` entry
- Install your package properly
- All imports will work!
