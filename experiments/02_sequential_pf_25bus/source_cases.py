"""Public benchmark inputs from bundled numeric snapshots, not source programs.

Read only bus/gen/branch arrays. Saved solutions are not used as Newton
references: those are recomputed from the input arrays with a separate solve.
"""

import p3_bootstrap as bootstrap
import numpy as np
from pypower.api import runpf, ppoption
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
from pypower.makeSbus import makeSbus


def exact_source_args():
    args = []
    references = []
    for name in ("Stagg5", "IEEE14"):
        path = bootstrap.HERE / "recorded_results" / f"{name}_exact_source_inputs.npz"
        with np.load(path) as snapshot:
            case = {key: snapshot[key].copy() for key in ("bus", "gen", "branch")}
        case.update(version="2", baseMVA=100)
        internal = ext2int(case)
        y = makeYbus(100, internal["bus"], internal["branch"])[0].tocsr()
        y.sort_indices()
        s = makeSbus(100, internal["bus"], internal["gen"])
        kind = internal["bus"][:, 1].astype(np.int64)
        vm = np.ones(len(kind))
        vm[internal["gen"][:, 0].astype(int)] = internal["gen"][:, 5]
        args.append(
            (
                y.indptr.astype(np.int64),
                y.indices.astype(np.int64),
                y.data,
                y.diagonal(),
                s,
                vm.astype(complex),
                kind,
                vm,
                1e-9,
                5000,
                8,
                1000.0,
            )
        )
        nr, ok = runpf(case, ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-11, PF_MAX_IT=50))
        assert ok, f"Independent Newton failed for {name}"
        references.append(nr["bus"][:, 7] * np.exp(1j * np.deg2rad(nr["bus"][:, 8])))
    return args, references
