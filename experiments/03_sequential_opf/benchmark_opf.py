"""Compare complete OPFs separately from identical fixed-work PF evaluations."""

import opf_bootstrap
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import random
import time
import json
import csv
import statistics
import numpy as np
from optimization import Evaluator, optimize, baseline
from opf_core import core_impl, core_native
from model import build_case

HERE = opf_bootstrap.HERE
OUT = HERE / "results"
SHARED_CASE = None
FIXED = None
ENGINE = None


def initializer(engine, barrier):
    global SHARED_CASE, FIXED, ENGINE
    SHARED_CASE = build_case()
    ENGINE = engine
    ev = Evaluator(engine=engine, base_case=SHARED_CASE)
    FIXED = ev.fixed
    if engine == "numba":
        core_native(*FIXED, baseline(), 1.0, 1, 1e-10, 1e-9)
    barrier.wait(timeout=120)


def solve_job(task, engine, base):
    # Each scenario owns its Evaluator/cache and optimizer arrays. The immutable
    # base case may be shared by threads; mutable PF histories are never shared.
    profile, start, free = task
    ev = Evaluator(float(profile), engine=engine, base_case=base)
    o = optimize(ev, np.asarray(start, float), free_frequency=bool(free))
    if not o["accepted"]:
        raise RuntimeError(
            f'Benchmark OPF failed: {o["message"]}; {o["result"].get("min_constraint")}'
        )
    r = o["result"]
    return dict(
        loss_mw=r["loss_mw"],
        x=r["x"].tolist(),
        min_constraint=r["min_constraint"],
        conservation_error_mw=r["power_balance_error_mw"],
        pf_evaluations=o["pf_evaluations"],
        iterations=o["iterations"],
        accepted=True,
    )


def process_opf(task):
    return solve_job(task, ENGINE, SHARED_CASE)


def pf_job(task, engine, fixed):
    # This workload evaluates one saved control vector, not an inference model.
    # It separates fixed PF work from optimizer line-search/iteration choices.
    profile, x = task
    fun = core_native if engine == "numba" else core_impl
    r = fun(*fixed, np.asarray(x, float), float(profile), 1, 1e-10, 1e-9)
    if not r[-1]:
        raise RuntimeError("Benchmark PF failure")
    return np.r_[r[0], r[1], r[2].astype(complex), r[3]], int(r[-2])


def process_pf(task):
    return pf_job(task, ENGINE, FIXED)


def make_runner(engine, execution, tasks, scope):
    if execution == "process":
        context = mp.get_context("spawn")
        barrier = context.Barrier(2)
        pool = ProcessPoolExecutor(
            max_workers=2,
            mp_context=context,
            initializer=initializer,
            initargs=(engine, barrier),
        )
        function = process_opf if scope == "complete_opf_peak_B_C" else process_pf
        return lambda: list(pool.map(function, tasks, chunksize=1)), pool
    base = build_case()
    fixed = Evaluator(base_case=base).fixed
    function = (
        (lambda task: solve_job(task, engine, base))
        if scope == "complete_opf_peak_B_C"
        else (lambda task: pf_job(task, engine, fixed))
    )
    if execution == "thread":
        pool = ThreadPoolExecutor(max_workers=2)
        return lambda: list(pool.map(function, tasks)), pool
    return lambda: [function(task) for task in tasks], None


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opf-repeats", type=int, default=3)
    parser.add_argument("--pf-repeats", type=int, default=7)
    opt = parser.parse_args()
    starts = json.loads((OUT / "optimization_starts.json").read_text())
    states = json.loads((OUT / "solution_controls.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    peak = starts[summary["peak_hour"]]
    opftasks = [
        (peak["profile"], peak["B_start"], False),
        (peak["profile"], peak["C_start"], True),
    ]
    pftasks = [(starts[s["hour"]]["profile"], s["x"]) for s in states]
    methods = [
        ("python", "process"),
        ("numba", "serial"),
        ("numba", "process"),
        ("numba", "thread"),
    ]
    samples = []
    aggregates = []
    firstcalls = []
    checks = []
    for scope, tasks, repeats in [
        ("complete_opf_peak_B_C", opftasks, opt.opf_repeats),
        ("fixed_work_72_solved_PF_points", pftasks, opt.pf_repeats),
    ]:
        base = build_case()
        fixed = Evaluator(base_case=base).fixed
        reference = (
            [solve_job(t, "numba", base) for t in tasks]
            if scope.startswith("complete")
            else [pf_job(t, "numba", fixed) for t in tasks]
        )
        runners = {}
        pools = []
        try:
            for engine, execution in methods:
                key = f"{engine}_{execution}"
                fn, pool = make_runner(engine, execution, tasks, scope)
                if pool is not None:
                    pools.append(pool)
                t = time.perf_counter()
                result = fn()
                setup = time.perf_counter() - t
                firstcalls.append(
                    dict(scope=scope, method=key, first_call_seconds=setup)
                )
                runners[key] = fn
                print("warmed", scope, key, round(setup, 3), "s", flush=True)
            for repeat in range(repeats):
                keys = list(runners)
                random.Random(441 + repeat).shuffle(keys)
                for key in keys:
                    t = time.perf_counter()
                    result = runners[key]()
                    elapsed = time.perf_counter() - t
                    if scope.startswith("complete"):
                        err = max(
                            abs(r["loss_mw"] - ref["loss_mw"])
                            for r, ref in zip(result, reference)
                        )
                        assert err < 1e-3 and all(
                            r["accepted"] and abs(r["conservation_error_mw"]) < 1e-5
                            for r in result
                        )
                        iterations = sum(r["iterations"] for r in result)
                        evals = sum(r["pf_evaluations"] for r in result)
                    else:
                        err = float(
                            max(
                                np.max(abs(r[0] - ref[0]))
                                for r, ref in zip(result, reference)
                            )
                        )
                        assert err < 1e-7
                        iterations = sum(r[1] for r in result)
                        evals = len(tasks)
                    samples.append(
                        dict(
                            scope=scope,
                            method=key,
                            repeat=repeat,
                            seconds=elapsed,
                            tasks=len(tasks),
                            pf_evaluations=evals,
                            iterations_or_gs_sweeps=iterations,
                            max_reference_error=err,
                            correctness_passed=True,
                        )
                    )
                    print(
                        "timed", scope, key, repeat, round(elapsed, 4), "s", flush=True
                    )
            pybase = statistics.median(
                r["seconds"]
                for r in samples
                if r["scope"] == scope and r["method"] == "python_process"
            )
            for key in runners:
                ss = [
                    r["seconds"]
                    for r in samples
                    if r["scope"] == scope and r["method"] == key
                ]
                med = statistics.median(ss)
                aggregates.append(
                    dict(
                        scope=scope,
                        method=key,
                        repeats=repeats,
                        tasks=len(tasks),
                        median_seconds=med,
                        min_seconds=min(ss),
                        max_seconds=max(ss),
                        p25_seconds=float(np.percentile(ss, 25)),
                        p75_seconds=float(np.percentile(ss, 75)),
                        ratio_vs_our_python_process=pybase / med,
                    )
                )
        finally:
            for pool in pools:
                pool.shutdown(wait=True)
    write_csv("benchmark_samples.csv", samples)
    write_csv("benchmark_summary.csv", aggregates)
    write_csv("benchmark_first_calls.csv", firstcalls)
    (OUT / "benchmark_protocol.json").write_text(
        json.dumps(
            dict(
                workers=2,
                native_library_threads=1,
                persistent_pools=True,
                preloaded_case_data=True,
                synchronized_process_initialization=True,
                excluded_JIT_and_pool_startup=True,
                opf_repeats=opt.opf_repeats,
                pf_repeats=opt.pf_repeats,
                opf_workload="Two independent peak-hour optimizations B and C, identical fixed starting vectors across backends.",
                pf_workload="Exactly72 already-solved hourly A/B/C control vectors; identical profiles, tolerances, numerical method and complete PF work.",
                compiled_scope="GS+Anderson AND converter/DC/outer PF coordination; optimizer remains SciPy SLSQP.",
                threads_parallelize="Independent scenario/OPF tasks; no parallelization of sequential GS bus updates.",
                no_ml_inference=True,
                authors_code_used=False,
                authors_exact_dataset=False,
                complete_opf_reference_error_units="MW loss objective; states can differ among equivalent local optima",
                fixed_work_reference_error_units="pu complex bus voltage or converter power",
                random_method_order=True,
                samples=len(samples),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
