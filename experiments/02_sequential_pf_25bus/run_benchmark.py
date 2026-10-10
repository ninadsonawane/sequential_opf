"""Warm, persistent-pool benchmarks. Never claim author-code speedup."""

import p3_bootstrap as bootstrap  # noqa: F401
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
import random
import statistics
import time
import numpy as np
from pypower.case14 import case14
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus
from pypower.makeSbus import makeSbus
from pypower.api import runpf, ppoption
from model import build_case, sequential, make_args, solve_impl, solve_native
from source_cases import exact_source_args

ARGS = None
OUT = bootstrap.HERE / "results"


def worker_init(args, engine, ready):
    global ARGS
    ARGS = args
    if engine == "numba":
        solve_native(*ARGS[0])
    ready.wait(timeout=120)


def worker_task(task):
    idx, engine = task
    return (solve_native if engine == "numba" else solve_impl)(*ARGS[idx])


def runner(args, engine, execution):
    fun = solve_native if engine == "numba" else solve_impl
    if execution == "process":
        context = mp.get_context("spawn")
        barrier = context.Barrier(2)
        pool = ProcessPoolExecutor(
            max_workers=2,
            mp_context=context,
            initializer=worker_init,
            initargs=(args, engine, barrier),
        )
        return (
            lambda: list(
                pool.map(worker_task, [(0, engine), (1, engine)], chunksize=1)
            ),
            pool,
        )
    if execution == "thread":
        pool = ThreadPoolExecutor(max_workers=2)
        return lambda: list(pool.map(lambda a: fun(*a), args)), pool
    return lambda: [fun(*a) for a in args], None


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def fingerprint(args):
    h = hashlib.sha256()
    for a in args:
        for x in a:
            h.update(x.tobytes() if isinstance(x, np.ndarray) else repr(x).encode())
    return h.hexdigest()


def verify_initial(result, reference):
    assert len(result) == len(reference) == 2
    for r, v in zip(result, reference):
        assert r[2] < 1e-9 and np.max(abs(r[0] - v)) < 1e-7


def verify_full(result, reference):
    error = max(np.max(abs(result[k] - reference[k])) for k in ("vh", "vl", "vd", "ss"))
    assert error < 1e-7, error
    assert result["trace"][-1]["dc_interface_mismatch_pu"] < 1e-8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=21)
    opt = parser.parse_args()
    case = build_case()
    args = [make_args(g, tol=1e-9) for g in case.initial_grids]
    sourceargs, sourcerefs = exact_source_args()
    reference = sequential(case)
    initial_ref = [solve_impl(*a)[0] for a in args]
    solve_native(*args[0])
    solve_native(*args[1])  # JIT outside warm measurements.
    scenarios = [
        ("exact_public_stagg5_ieee14_pair", sourceargs, sourcerefs, False),
        ("reconstructed_initial_hvac_5_14", args, initial_ref, False),
        ("reconstructed_full_25bus", args, initial_ref, True),
    ]
    rows = []
    summary = []
    pools = []
    try:
        for scope, inputs, refs, full in scenarios:
            hashes_before = fingerprint(inputs)
            methods = {}
            for engine in ("python", "numba"):
                for execution in ("serial", "process", "thread"):
                    key = f"{engine}_{execution}"
                    initial, pool = runner(inputs, engine, execution)
                    if pool is not None:
                        pools.append(pool)
                    if full:
                        fn = lambda e=engine, ini=initial: sequential(
                            case, engine=e, initial_runner=ini
                        )
                        verify = lambda result: verify_full(result, reference)
                    else:
                        fn = initial
                        verify = lambda result, rr=refs: verify_initial(result, rr)
                    # This call blocks until every process has imported and (if
                    # needed) loaded the compiled function; not in warm timing.
                    warm_start = time.perf_counter()
                    result = fn()
                    warm = time.perf_counter() - warm_start
                    verify(result)
                    methods[key] = (fn, verify, engine, execution, warm)
            schedule = list(methods) * opt.repeats
            random.Random(20261009 + len(rows)).shuffle(schedule)
            for n, key in enumerate(schedule):
                fn, verify, engine, execution, warm = methods[key]
                start = time.perf_counter()
                result = fn()
                elapsed = time.perf_counter() - start
                verify(result)  # Correctness check outside timing.
                iterations = (
                    result["total_sweeps"] if full else sum(r[1] for r in result)
                )
                rows.append(
                    dict(
                        scope=scope,
                        method=key,
                        engine=engine,
                        execution=execution,
                        sample_order=n,
                        seconds=elapsed,
                        total_gs_sweeps=int(iterations),
                        max_reference_error_pu=(
                            float(
                                max(
                                    np.max(abs(result[k] - reference[k]))
                                    for k in ("vh", "vl", "vd", "ss")
                                )
                            )
                            if full
                            else float(
                                max(np.max(abs(r[0] - v)) for r, v in zip(result, refs))
                            )
                        ),
                        correctness_passed=True,
                    )
                )
            assert hashes_before == fingerprint(inputs), "Solver modified its inputs"
            baseline = statistics.median(
                r["seconds"]
                for r in rows
                if r["scope"] == scope and r["method"] == "python_process"
            )
            for key, (_, _, engine, execution, warm) in methods.items():
                samples = [
                    r["seconds"]
                    for r in rows
                    if r["scope"] == scope and r["method"] == key
                ]
                med = statistics.median(samples)
                summary.append(
                    dict(
                        scope=scope,
                        method=key,
                        engine=engine,
                        execution=execution,
                        repeats=len(samples),
                        median_ms=med * 1000,
                        p25_ms=float(np.percentile(samples, 25) * 1000),
                        p75_ms=float(np.percentile(samples, 75) * 1000),
                        min_ms=min(samples) * 1000,
                        max_ms=max(samples) * 1000,
                        ratio_vs_our_python_process=baseline / med,
                        pool_setup_first_call_ms=warm * 1000,
                    )
                )
            print(
                scope,
                [
                    (r["method"], round(r["median_ms"], 3))
                    for r in summary
                    if r["scope"] == scope
                ],
                flush=True,
            )
            # Each scenario has its own initialized input; end its workers after
            # all methods are measured. No worker startup included in medians.
            for pool in pools:
                pool.shutdown(wait=True)
            pools = []
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    save_csv(OUT / "timing_samples.csv", rows)
    save_csv(OUT / "timing_summary.csv", summary)
    (OUT / "benchmark_protocol.json").write_text(
        json.dumps(
            dict(
                workers=2,
                repeats=opt.repeats,
                randomized_method_order=True,
                persistent_pools=True,
                preloaded_process_inputs=True,
                all_workers_synchronized_before_timing=True,
                native_library_threads=1,
                full_solver_parallel_stage="only initial independent HVAC solves, Fig5 step1",
                full_solver_remaining_stages="coupled HVAC/LFAC solves, VSC updates and DC coordination run serially",
                input_sha256=fingerprint(args),
                algorithm="same solve_impl Python/Numba function; GS+Anderson depth8",
                initialization="flat per independent grid, then initial solutions for coupled HVAC stage",
                inner_tolerance=1e-9,
                outer_tolerance=1e-8,
                exclude_JIT_and_pool_startup=True,
                start_method="spawn",
                numba_flags={"cache": True, "nogil": True, "fastmath": False},
                authors_code_used=False,
                paper_full_case_exact=False,
                exact_public_cases_used=[
                    "unmodified Stagg5 numeric snapshot",
                    "unmodified IEEE14 numeric snapshot",
                ],
                source_programs_vendored=False,
                samples=len(rows),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
