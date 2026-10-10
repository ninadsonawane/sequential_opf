# 1. Compilation and execution benchmark

**Question:** can the same Gauss–Seidel + Anderson algorithm run faster when
compiled, and when independent grids are distributed to threads or processes?

`benchmark.py` defines one `solve_impl` function and compiles that same function
with `njit(cache=True, nogil=True, fastmath=False)`. Both paths retain in-place
bus ordering, PV magnitude projection, complex phasors, depth-eight Anderson
history and the same P/Q residual test. No dependent bus updates are parallelized.

## Reproduce

From the repository root with the documented environment installed:

```bash
# Accuracy only: IEEE 14/39/118, Python/Numba/plain GS versus Newton
python experiments/01_numba_benchmark/benchmark.py --smoke --output experiments/01_numba_benchmark/results
# Complete timing study
python experiments/01_numba_benchmark/benchmark.py --workers 4 --repeats 7 --output experiments/01_numba_benchmark/results
```

Inputs come from installed PYPOWER, not a local `_deps` folder or network fetch.
The run exports input/state arrays for each unique IEEE case as NPZ. Committed
baseline inputs and reports are in [recorded_results](recorded_results/report.md).

## Workloads and interpretation

- 590 AC buses: five independent IEEE118 replicas.
- 1180 AC buses: ten IEEE118 replicas.
- 1960 AC buses: ten IEEE118 plus twenty IEEE39 replicas.
- Separate equal-work control: exactly 100 plain GS sweeps in Python and Numba.

These resemble the sizes of P3's initial independent AC stage, not its complete
coupled large networks. Converter, tie-line and DC iterations are absent.
Compare Numba serial to Python serial for compilation, then compiled pools to
Numba serial for parallel benefit. Pool overhead can dominate small tasks.

Warm pools persist; process inputs are preloaded and workers synchronized before
measurement; method order is randomized. One native-library thread avoids nested
oversubscription. All final voltage vectors are returned and checked outside
timing. First invocations/JIT are recorded separately. PV Q-limit switching is
outside this kernel experiment. Floating-point history mixing can change AA
iteration counts; the physical solution and tolerances are checked instead.

**Outputs:** `validation.json` in smoke mode; full mode adds `timings.csv`,
`results.json`, `comparison.svg/.png`, `report.md` and `ieee*_inputs.npz`.

**Next:** [complete 25-bus hybrid power flow](../02_sequential_pf_25bus/README.md).
