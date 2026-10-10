"""Select sibling research modules, never machine-local dependency folders."""

from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
for name in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl_cache"))
sys.path.insert(0, str(HERE.parent / "01_numba_benchmark"))
sys.path.insert(0, str(HERE))
