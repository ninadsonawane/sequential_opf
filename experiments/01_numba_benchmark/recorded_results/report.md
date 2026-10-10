# P3-inspired Numba / multiprocessing experiment

**Completed stage:** independent initial AC per-grid power flow from P3 Fig. 5 / Section 5.2.4.
**Not an exact reproduction:** authors state data available on request; their assembled multi-frequency networks and implementation were not available. No tie-line or VSC/DC solve is timed.

## Measured results

| Configuration | Method | Median (ms) | Speedup vs Python processes |
|---|---|---:|---:|
| 590 AC buses / 5 grids | Python serial | 1412.465 | 0.45x |
| 590 AC buses / 5 grids | Python threads | 1047.614 | 0.61x |
| 590 AC buses / 5 grids | Python processes | 636.642 | 1.00x |
| 590 AC buses / 5 grids | Numba serial | 55.498 | 11.47x |
| 590 AC buses / 5 grids | Numba threads | 21.868 | 29.11x |
| 590 AC buses / 5 grids | Numba processes | 24.677 | 25.80x |
| 1180 AC buses / 10 grids | Python serial | 2169.906 | 0.32x |
| 1180 AC buses / 10 grids | Python threads | 2207.663 | 0.31x |
| 1180 AC buses / 10 grids | Python processes | 686.135 | 1.00x |
| 1180 AC buses / 10 grids | Numba serial | 81.907 | 8.38x |
| 1180 AC buses / 10 grids | Numba threads | 30.811 | 22.27x |
| 1180 AC buses / 10 grids | Numba processes | 31.549 | 21.75x |
| 1960 AC buses / 30 grids | Python serial | 2197.049 | 0.37x |
| 1960 AC buses / 30 grids | Python threads | 2277.269 | 0.36x |
| 1960 AC buses / 30 grids | Python processes | 819.324 | 1.00x |
| 1960 AC buses / 30 grids | Numba serial | 134.387 | 6.10x |
| 1960 AC buses / 30 grids | Numba threads | 40.293 | 20.33x |
| 1960 AC buses / 30 grids | Numba processes | 42.449 | 19.30x |

## Algorithm and accuracy

- Python and Numba run the same function; sparse CSR admittance, sequential in-place GS, PV Q calculation/magnitude projection, Anderson history depth 8 and gamma threshold 1000.
- All runs use complex128/float64, no fastmath, flat-angle initialization, P/Q mismatch tolerance 1e-6 pu, and max 3000 iterations.
- PV reactive limits are not enforced (fixed PV classification). These are standard benchmark operating points, not a study of limit-induced mode switching.
- Anderson is applied to complex phasors; PV magnitudes are restored after mixing. The paper does not publish every implementation detail (gamma threshold/order); these choices are explicit.
- Newton-Raphson at 1e-10 pu independently verifies the physical solution. Iteration count equivalence is recorded rather than assumed.
- Python/Numba AA iteration counts differ slightly: compiled arithmetic and least-squares rounding affect residual-history mixing. Mathematical algorithm and tolerances are preserved, but the iterate sequence is not bitwise identical.

| Case | Python AA iterations | Numba AA iterations | Plain GS iterations | Max phasor difference | NR difference |
|---|---:|---:|---:|---:|---:|
| IEEE 14 | 17 | 17 | 179 | 1.343e-15 | 9.005e-08 |
| IEEE 39 | 105 | 116 | 1846 | 3.662e-08 | 4.027e-08 |
| IEEE 118 | 173 | 181 | 2000 | 3.440e-07 | 1.291e-07 |

## Timing protocol

- 7 measured batches after one warm batch per method/configuration; median, min/max and raw repeats saved.
- Four executor workers; all BLAS/OpenMP/Numba internal thread counts set to one.
- Processes persist during repeats, inputs reside in worker memory, and final voltage vectors return to parent.
- An initializer barrier ensures ALL process workers finish imports and native warm-up before any batch runs. This prevents worker startup leaking into fast Numba timings.
- Thread workers share immutable inputs; each solver owns its voltage/history arrays. Numba compiles the full solver with nogil=True; no dependent GS bus updates are parallelized.
- First-call JIT/cache loading and process startup are recorded separately in results.json. Each task solves one grid; no artificial repetition is added inside a task.
- Methods are interleaved in a seeded random order in each repeat; runtime is sensitive to CPU scheduling/thermal state. Do not equate these timings with the paper's machine or its complete coupled PF timings.

## Interpretation

Compilation and parallel execution are different effects. Compare Numba serial against Python serial for compilation; compare Numba threads/processes against Numba serial for added parallel benefit. Fast compiled tasks can be dominated by executor overhead.
A speedup of the initial AC stage does not establish a speedup of the complete multi-frequency algorithm.

## Equal-work compilation control

To isolate compilation from differences in Anderson iteration counts, both implementations also execute exactly 100 plain GS sweeps from the same input. This is a kernel timing control, not a converged PF solve.

| Case | Python (ms) | Numba (ms) | Speedup | Max phasor difference |
|---|---:|---:|---:|---:|
| IEEE 39 | 153.612 | 0.604 | 254.28x | 4.495e-16 |
| IEEE 118 | 512.326 | 2.023 | 253.31x | 3.469e-16 |

## Reproduction

Run `benchmark.py --workers 4 --repeats 7`. Dependencies are NumPy, SciPy, Numba, PYPOWER, Matplotlib.
Inputs/results for each unique IEEE case are saved as NPZ files. Full environment and code SHA256 are in results.json.

## Sources

- P3: https://doi.org/10.1016/j.ijepes.2026.112025 (Sections 4.1, 4.3, 5.2.3, 5.2.4; Tables 9-11).
- IEEE benchmark data distributed by PYPOWER: https://github.com/rwl/PYPOWER/tree/master/pypower
- Numba GIL/compilation: https://numba.readthedocs.io/en/stable/user/jit.html

The 1960-AC-bus configuration is the AC portion of P3's 1966-bus case: 10 x IEEE118 + 20 x IEEE39. Its 3 DC + 3 LF-AC buses and coupled solve are excluded.
The 51,860-bus system is not reproduced; obtaining the assembled data is required for its topology-specific test.