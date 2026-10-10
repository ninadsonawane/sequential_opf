"""P3-inspired per-grid GS + Anderson benchmark; not the coupled tie-line PF.

Python and Numba execute the SAME solve_impl function. No fastmath, prange,
bus-update reordering, changed tolerance, or replacement of GS by Newton.
"""

from pathlib import Path
import os

# Limit native library threads before importing NumPy in parent and workers.
for key in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[key] = "1"

import sys

HERE = Path(__file__).resolve().parent
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import platform
import random
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import scipy
import numba
from numba import njit
from pypower.case14 import case14
from pypower.case39 import case39
from pypower.case118 import case118
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
from pypower.makeSbus import makeSbus
from pypower.runpf import runpf
from pypower.ppoption import ppoption


def solve_impl(
    ptr, cols, vals, diagonal, specified, v0, kind, vm, tol, maxiter, depth, gamma_max
):
    """CSR in-place GS, projected PV voltages, Algorithm-1 Anderson mixing.

    Natural bus order; each PV Q is recomputed using the newest available V.
    AA mixes complex phasors and restores PV magnitudes. Convergence is
    checked against P mismatch (PV/PQ) and Q mismatch (PQ), not only dV.
    """
    n = len(v0)
    v = v0.copy()
    hx = np.empty((depth + 1, n), dtype=np.complex128)
    hf = np.empty((depth + 1, n), dtype=np.complex128)
    count = 0
    residual = 1e100
    fallbacks = 0
    for iteration in range(1, maxiter + 1):
        old = v.copy()
        for i in range(n):
            if kind[i] == 3:
                continue
            current = 0.0j
            for j in range(ptr[i], ptr[i + 1]):
                current += vals[j] * v[cols[j]]
            s = specified[i]
            if kind[i] == 2:
                s = complex(s.real, (v[i] * current.conjugate()).imag)
            v[i] += ((s / v[i]).conjugate() - current) / diagonal[i]
            if kind[i] == 2:
                v[i] *= vm[i] / abs(v[i])
        fixed_residual = v - old
        if depth > 0:
            if count == depth + 1:
                for row in range(depth):
                    hx[row, :] = hx[row + 1, :]
                    hf[row, :] = hf[row + 1, :]
                count = depth
            hx[count, :] = old
            hf[count, :] = fixed_residual
            count += 1
            m = count - 1
            if m > 0:
                dx = np.empty((n, m), dtype=np.complex128)
                df = np.empty((n, m), dtype=np.complex128)
                for col in range(m):
                    dx[:, col] = hx[col + 1, :] - hx[col, :]
                    df[:, col] = hf[col + 1, :] - hf[col, :]
                gamma = np.linalg.lstsq(df, fixed_residual, rcond=-1.0)[0]
                if np.max(np.abs(gamma)) > gamma_max:
                    count = 0
                    fallbacks += 1
                else:
                    v = old + fixed_residual - (dx + df) @ gamma
                    for i in range(n):
                        if kind[i] == 2:
                            v[i] *= vm[i] / abs(v[i])
                        elif kind[i] == 3:
                            v[i] = v0[i]
        residual = 0.0
        for i in range(n):
            if kind[i] == 3:
                continue
            current = 0.0j
            for j in range(ptr[i], ptr[i + 1]):
                current += vals[j] * v[cols[j]]
            mismatch = v[i] * current.conjugate() - specified[i]
            residual = max(residual, abs(mismatch.real))
            if kind[i] == 1:
                residual = max(residual, abs(mismatch.imag))
        if not np.isfinite(np.max(np.abs(v))):
            return v, iteration, np.inf, fallbacks
        if residual < tol:
            return v, iteration, residual, fallbacks
    return v, maxiter, residual, fallbacks


solve_native = njit(cache=True, nogil=True, fastmath=False)(solve_impl)


def make_case(size, tol=1e-6, depth=8):
    external = {14: case14, 39: case39, 118: case118}[size]()
    case = ext2int(external)
    y, _, _ = makeYbus(case["baseMVA"], case["bus"], case["branch"])
    y = y.tocsr()
    y.sort_indices()
    s = makeSbus(case["baseMVA"], case["bus"], case["gen"])
    kind = np.asarray(case["bus"][:, 1], dtype=np.int64)
    vm = np.ones(size)
    for gen in case["gen"]:
        if gen[7] > 0:
            vm[int(gen[0])] = gen[5]
    v0 = vm.astype(np.complex128)  # Flat angles; specified PV/slack magnitudes.
    args = (
        y.indptr.astype(np.int64),
        y.indices.astype(np.int64),
        y.data.astype(np.complex128),
        y.diagonal().astype(np.complex128),
        np.asarray(s, dtype=np.complex128),
        v0,
        kind,
        vm,
        tol,
        3000,
        depth,
        1000.0,
    )
    reference, success = runpf(
        external, ppoption(PF_TOL=1e-10, VERBOSE=0, OUT_ALL=0, PF_ALG=1)
    )
    if not success:
        raise RuntimeError(f"Newton reference failed: IEEE {size}")
    nr = reference["bus"][:, 7] * np.exp(1j * np.deg2rad(reference["bus"][:, 8]))
    # IEEE118's published reference angle is nonzero; our flat-angle start
    # uses zero. Align the independent NR result to this same reference.
    ref = int(np.flatnonzero(kind == 3)[0])
    nr *= np.exp(1j * (np.angle(v0[ref]) - np.angle(nr[ref])))
    return args, nr, external


WORKER_CASES = None


def worker_init(cases, engine, ready_barrier=None):
    global WORKER_CASES
    WORKER_CASES = cases
    if engine == "numba":
        solve_native(*cases[0])  # JIT/cache loading excluded from warm timings.
    # A tiny compiled batch could finish on one early worker while other
    # workers are still importing/JIT-loading. Synchronize ALL workers first.
    if ready_barrier is not None:
        ready_barrier.wait(timeout=120)


def worker_task(task):
    idx, engine = task
    return (solve_native if engine == "numba" else solve_impl)(*WORKER_CASES[idx])


def make_runner(engine, execution, cases, workers):
    if execution == "process":
        context = mp.get_context("spawn")
        ready = context.Barrier(workers)
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=worker_init,
            initargs=(cases, engine, ready),
        )
        tasks = [(i, engine) for i in range(len(cases))]
        return lambda: list(pool.map(worker_task, tasks, chunksize=1)), pool
    function = solve_native if engine == "numba" else solve_impl
    if execution == "thread":
        pool = ThreadPoolExecutor(max_workers=workers)
        return lambda: list(pool.map(lambda a: function(*a), cases)), pool
    return lambda: [function(*a) for a in cases], None


def check_results(results, refs, cases):
    max_delta = max_nr = max_mismatch = 0.0
    iterations = []
    for out, ref, args in zip(results, refs, cases):
        v, it, mismatch, fallback = out
        if not np.all(np.isfinite(v)) or mismatch >= args[8]:
            raise AssertionError(f"Nonconverged PF: it={it}, residual={mismatch}")
        max_mismatch = max(max_mismatch, float(mismatch))
        max_nr = max(max_nr, float(np.max(np.abs(v - ref["nr"]))))
        max_delta = max(max_delta, float(np.max(np.abs(v - ref["python"]))))
        if max_nr > 2e-5 or max_delta > 2e-5:
            raise AssertionError(f"Voltage verification failed: {max_nr}, {max_delta}")
        iterations.append(int(it))
    return max_delta, max_nr, max_mismatch, iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=None)
    opt = parser.parse_args()
    # Explicit output paths are relative to the invoking working directory.
    # With no override, keep outputs alongside this experiment's script.
    output = Path(opt.output).resolve() if opt.output else HERE / "results"
    output.mkdir(parents=True, exist_ok=True)
    cases_by_size = {}
    refs_by_size = {}
    checks = []
    for size in (14, 39, 118):
        args, nr, external = make_case(size)
        py = solve_impl(*args)
        native_start = time.perf_counter()
        native = solve_native(*args)
        native_first_s = time.perf_counter() - native_start
        refs = {"python": py[0], "nr": nr}
        delta, nr_delta, mismatch, its = check_results([native], [refs], [args])
        check_results([py], [refs], [args])
        plain = solve_native(*args[:10], 0, args[11])
        check_results([plain], [refs], [args])
        entry = dict(
            case=f"IEEE {size}",
            python_iterations=int(py[1]),
            numba_iterations=int(native[1]),
            plain_gs_iterations=int(plain[1]),
            max_python_numba_phasor_delta=delta,
            max_nr_phasor_delta=nr_delta,
            max_mismatch_pu=mismatch,
            native_first_call_s=native_first_s,
            aa_fallbacks=int(native[3]),
        )
        checks.append(entry)
        cases_by_size[size] = args
        refs_by_size[size] = refs
        np.savez(
            output / f"ieee{size}_inputs.npz",
            **{
                name: val
                for name, val in zip(
                    (
                        "indptr",
                        "indices",
                        "admittance",
                        "diagonal",
                        "Sbus",
                        "V0",
                        "bus_type",
                        "Vm",
                    ),
                    args[:8],
                )
            },
            newton_voltage=nr,
            python_voltage=py[0],
            numba_voltage=native[0],
        )
        print("VALIDATION", json.dumps(entry), flush=True)
    if opt.smoke:
        (output / "validation.json").write_text(json.dumps(checks, indent=2))
        return

    # Equal-work control: 100 unaccelerated GS sweeps in both implementations.
    # This isolates compilation from AA's floating-point-sensitive iteration count.
    fixed_sweeps = []
    for size in (39, 118):
        args = cases_by_size[size]
        fixed = (*args[:8], -1.0, 100, 0, args[11])
        py = solve_impl(*fixed)
        native = solve_native(*fixed)
        delta = float(np.max(np.abs(py[0] - native[0])))
        if delta > 1e-10 or py[1] != 100 or native[1] != 100:
            raise AssertionError("Fixed-sweep equivalence failed")
        timings = {}
        for name, fn in [("Python", solve_impl), ("Numba", solve_native)]:
            samples = []
            for _ in range(opt.repeats):
                start = time.perf_counter()
                fn(*fixed)
                samples.append(time.perf_counter() - start)
            timings[name] = statistics.median(samples)
        fixed_sweeps.append(
            dict(
                case=f"IEEE {size}",
                sweeps=100,
                python_s=timings["Python"],
                numba_s=timings["Numba"],
                speedup=timings["Python"] / timings["Numba"],
                max_phasor_delta=delta,
            )
        )

    # Same AC subproblem sizes/replica counts as Table 11 and the AC part of 5.2.3.
    configs = [
        ("590 AC buses / 5 grids", [118] * 5),
        ("1180 AC buses / 10 grids", [118] * 10),
        ("1960 AC buses / 30 grids", [118] * 10 + [39] * 20),
    ]
    modes = [
        ("Python serial", "python", "serial"),
        ("Python threads", "python", "thread"),
        ("Python processes", "python", "process"),
        ("Numba serial", "numba", "serial"),
        ("Numba threads", "numba", "thread"),
        ("Numba processes", "numba", "process"),
    ]
    raw = []
    summaries = []
    startup_rows = []
    for config, sizes in configs:
        cases = [cases_by_size[s] for s in sizes]
        refs = [refs_by_size[s] for s in sizes]
        runners = {}
        pools = []
        samples = {label: [] for label, _, _ in modes}
        last_checks = {}
        try:
            for label, engine, execution in modes:
                setup_start = time.perf_counter()
                invoke, pool = make_runner(engine, execution, cases, opt.workers)
                runners[label] = invoke
                if pool is not None:
                    pools.append(pool)
                warm = invoke()
                warm_s = time.perf_counter() - setup_start
                check_results(warm, refs, cases)
                startup_rows.append(
                    dict(config=config, method=label, setup_and_first_batch_s=warm_s)
                )
            rng = random.Random(42 + sum(sizes))
            for rep in range(opt.repeats):
                order = list(runners)
                rng.shuffle(order)
                for label in order:
                    start = time.perf_counter()
                    results = runners[label]()
                    seconds = time.perf_counter() - start
                    delta, nr_delta, mismatch, its = check_results(results, refs, cases)
                    samples[label].append(seconds)
                    last_checks[label] = (delta, nr_delta, mismatch, its)
                    raw.append(
                        dict(
                            config=config,
                            method=label,
                            repeat=rep,
                            seconds=seconds,
                            voltage_delta=delta,
                            nr_delta=nr_delta,
                            mismatch_pu=mismatch,
                            iterations=json.dumps(its),
                        )
                    )
                print("ROUND", config, rep + 1, "of", opt.repeats, flush=True)
            for label, engine, execution in modes:
                delta, nr_delta, mismatch, its = last_checks[label]
                summary = dict(
                    config=config,
                    method=label,
                    buses=sum(sizes),
                    grids=len(sizes),
                    workers=1 if execution == "serial" else opt.workers,
                    median_s=statistics.median(samples[label]),
                    min_s=min(samples[label]),
                    max_s=max(samples[label]),
                    max_voltage_delta=delta,
                    max_nr_delta=nr_delta,
                    max_mismatch_pu=mismatch,
                    iterations=its,
                )
                summaries.append(summary)
                print("BENCHMARK", json.dumps(summary), flush=True)
        finally:
            for pool in pools:
                pool.shutdown(wait=True)
    for row in summaries:
        group = [x for x in summaries if x["config"] == row["config"]]
        pyserial = next(x["median_s"] for x in group if x["method"] == "Python serial")
        pyprocess = next(
            x["median_s"] for x in group if x["method"] == "Python processes"
        )
        row["speedup_vs_python_serial"] = pyserial / row["median_s"]
        row["speedup_vs_python_processes"] = pyprocess / row["median_s"]
    with (output / "timings.csv").open("w", newline="") as file:
        w = csv.DictWriter(file, fieldnames=list(raw[0]))
        w.writeheader()
        w.writerows(raw)
    metadata = dict(
        python=sys.version,
        numpy=np.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
        platform=platform.platform(),
        cpu=os.environ.get("BENCHMARK_CPU_NAME", platform.processor()),
        logical_cpus=os.cpu_count(),
        workers=opt.workers,
        repeats=opt.repeats,
        tolerance=1e-6,
        max_iterations=3000,
        history_depth=8,
        gamma_max=1000,
        rng="none; unmodified standard cases, identical replicas",
        blas_threads=1,
        numba_nopython_signatures=str(solve_native.nopython_signatures),
        solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Independent initial AC per-grid PF stage only; no converters or tie-line iteration.",
        timings="Warm pools; preloaded inputs; all final voltages returned; includes dispatch/IPC.",
        first_calls=checks,
        startup=startup_rows,
    )
    metadata["fixed_sweep_control"] = fixed_sweeps
    (output / "results.json").write_text(
        json.dumps(
            dict(metadata=metadata, validation=checks, results=summaries), indent=2
        )
    )
    create_report(output, summaries, metadata, checks)


def create_report(output, rows, metadata, checks):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "Python serial",
        "Python threads",
        "Python processes",
        "Numba serial",
        "Numba threads",
        "Numba processes",
    ]
    configs = list(dict.fromkeys(x["config"] for x in rows))
    fig, axes = plt.subplots(1, len(configs), figsize=(15, 5), layout="constrained")
    for ax, config in zip(axes, configs):
        group = [
            next(x for x in rows if x["config"] == config and x["method"] == lab)
            for lab in labels
        ]
        ax.bar(
            range(6),
            [x["median_s"] * 1000 for x in group],
            color=["#8298ac"] * 3 + ["#168378"] * 3,
        )
        ax.set_yscale("log")
        ax.set_xticks(range(6), [x.replace(" ", "\n") for x in labels], fontsize=8)
        ax.set_title(config)
        ax.set_ylabel("Median batch time (ms, logarithmic scale)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"P3-inspired initial per-grid PF: same GS + Anderson equations, {metadata['workers']} workers\n"
        "Independent AC stage; excludes coupled tie-line/converter solution",
        fontsize=12,
    )
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    lines = [
        "# P3-inspired Numba / multiprocessing experiment",
        "",
        "**Completed stage:** independent initial AC per-grid power flow from P3 Fig. 5 / Section 5.2.4.",
        "**Not an exact reproduction:** authors state data available on request; their assembled "
        "multi-frequency networks and implementation were not available. No tie-line or VSC/DC solve is timed.",
        "",
        "## Measured results",
        "",
        "| Configuration | Method | Median (ms) | Speedup vs Python processes |",
        "|---|---|---:|---:|",
    ]
    for x in rows:
        lines.append(
            f"| {x['config']} | {x['method']} | {1000*x['median_s']:.3f} | {x['speedup_vs_python_processes']:.2f}x |"
        )
    lines += [
        "",
        "## Algorithm and accuracy",
        "",
        "- Python and Numba run the same function; sparse CSR admittance, sequential in-place GS, "
        "PV Q calculation/magnitude projection, Anderson history depth 8 and gamma threshold 1000.",
        "- All runs use complex128/float64, no fastmath, flat-angle initialization, "
        "P/Q mismatch tolerance 1e-6 pu, and max 3000 iterations.",
        "- PV reactive limits are not enforced (fixed PV classification). These are standard "
        "benchmark operating points, not a study of limit-induced mode switching.",
        "- Anderson is applied to complex phasors; PV magnitudes are restored after mixing. "
        "The paper does not publish every implementation detail (gamma threshold/order); these choices are explicit.",
        "- Newton-Raphson at 1e-10 pu independently verifies the physical solution. "
        "Iteration count equivalence is recorded rather than assumed.",
        "- Python/Numba AA iteration counts differ slightly: compiled arithmetic and "
        "least-squares rounding affect residual-history mixing. Mathematical algorithm and "
        "tolerances are preserved, but the iterate sequence is not bitwise identical.",
        "",
        "| Case | Python AA iterations | Numba AA iterations | Plain GS iterations | Max phasor difference | NR difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in checks:
        lines.append(
            f"| {x['case']} | {x['python_iterations']} | {x['numba_iterations']} | {x['plain_gs_iterations']} | {x['max_python_numba_phasor_delta']:.3e} | {x['max_nr_phasor_delta']:.3e} |"
        )
    lines += [
        "",
        "## Timing protocol",
        "",
        f"- {metadata['repeats']} measured batches after one warm batch per method/configuration; median, min/max and raw repeats saved.",
        f"- {metadata['workers']} executor workers; all BLAS/OpenMP/Numba internal thread counts set to one.",
        "- Processes persist during repeats, inputs reside in worker memory, and final voltage vectors return to parent.",
        "- An initializer barrier ensures ALL process workers finish imports and native warm-up "
        "before any batch runs. This prevents worker startup leaking into fast Numba timings.",
        "- Thread workers share immutable inputs; each solver owns its voltage/history arrays. "
        "Numba compiles the full solver with nogil=True; no dependent GS bus updates are parallelized.",
        "- First-call JIT/cache loading and process startup are recorded separately in results.json. "
        "Each task solves one grid; no artificial repetition is added inside a task.",
        "- Methods are interleaved in a seeded random order in each repeat; runtime is sensitive to CPU scheduling/thermal state. "
        "Do not equate these timings with the paper's machine or its complete coupled PF timings.",
        "",
        "## Interpretation",
        "",
        "Compilation and parallel execution are different effects. Compare Numba serial against "
        "Python serial for compilation; compare Numba threads/processes against Numba serial "
        "for added parallel benefit. Fast compiled tasks can be dominated by executor overhead.",
        "A speedup of the initial AC stage does not establish a speedup of the complete multi-frequency algorithm.",
        "",
        "## Equal-work compilation control",
        "",
        "To isolate compilation from differences in Anderson iteration counts, both implementations "
        "also execute exactly 100 plain GS sweeps from the same input. This is a kernel timing "
        "control, not a converged PF solve.",
        "",
        "| Case | Python (ms) | Numba (ms) | Speedup | Max phasor difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in metadata["fixed_sweep_control"]:
        lines.append(
            f"| {x['case']} | {x['python_s']*1000:.3f} | {x['numba_s']*1000:.3f} | {x['speedup']:.2f}x | {x['max_phasor_delta']:.3e} |"
        )
    lines += [
        "",
        "## Reproduction",
        "",
        "Run `benchmark.py --workers 4 --repeats 7`. Dependencies are NumPy, SciPy, Numba, PYPOWER, Matplotlib.",
        "Inputs/results for each unique IEEE case are saved as NPZ files. Full environment and code SHA256 are in results.json.",
        "",
        "## Sources",
        "",
        "- P3: https://doi.org/10.1016/j.ijepes.2026.112025 (Sections 4.1, 4.3, 5.2.3, 5.2.4; Tables 9-11).",
        "- IEEE benchmark data distributed by PYPOWER: https://github.com/rwl/PYPOWER/tree/master/pypower",
        "- Numba GIL/compilation: https://numba.readthedocs.io/en/stable/user/jit.html",
        "",
        "The 1960-AC-bus configuration is the AC portion of P3's 1966-bus case: "
        "10 x IEEE118 + 20 x IEEE39. Its 3 DC + 3 LF-AC buses and coupled solve are excluded.",
        "The 51,860-bus system is not reproduced; obtaining the assembled data is "
        "required for its topology-specific test.",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
