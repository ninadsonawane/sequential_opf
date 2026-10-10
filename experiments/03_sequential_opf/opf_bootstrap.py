"""Resolve sibling modules and use dependencies installed in the active venv."""

from pathlib import Path
import sys
import os

HERE = Path(__file__).resolve().parent
for name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl_cache"))
sys.path.insert(0, str(HERE.parent / "02_sequential_pf_25bus"))
import p3_bootstrap

sys.path.insert(0, str(HERE))
