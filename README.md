# Sequential power flow and OPF — research experiments

Three progressive experiments investigate whether compiling a sequential
power-flow solver improves the computational cost of hybrid-grid analysis and
loss-minimizing operation. They move from a numerical kernel to a complete
reconstructed network, then to constrained optimization.

**Research status:** independently implemented experiments inspired by
[Kim, Nguyen & Santoso, 2026 (P3)](https://doi.org/10.1016/j.ijepes.2026.112025)
and [Nguyen & Santoso, 2021 (P1)](https://doi.org/10.1109/TPWRS.2020.3037967).
The authors' complete modified case inputs and source code were not available.
This repository supplies **our disclosed numerical reconstruction**, not an
exact reproduction of either paper. Reported speedups compare our backends.

## Progression

| Stage | Research question | Electrical / computational scope | Start here |
|---|---|---|---|
| 1 | Does compilation preserve the GS–Anderson solution and reduce runtime? | Independent IEEE 14/39/118 AC solves; serial, threads and processes; replicated-grid batches | [Numba experiment](experiments/01_numba_benchmark/README.md) |
| 2 | Can the complete reconstructed hybrid circuit be solved and checked? | 19 HVAC + 3 LFAC + 3 DC buses, seven VSCs, AC tie and B2B/DC coordination | [25-bus sequential PF](experiments/02_sequential_pf_25bus/README.md) |
| 3 | Can sequential PF evaluate feasible planning and operating decisions? | Reduced-space continuous SLSQP; voltage/frequency planning, dispatch and hourly loss comparisons | [Sequential OPF](experiments/03_sequential_opf/README.md) |

Gauss–Seidel and Anderson solve the **state equations**. SLSQP chooses the
**controls**. Numba changes execution speed, not the objective or physical
equations. Threads distribute independent problems; dependent GS bus updates
are not parallelized.

## Fresh clone

Use **Python 3.12**, Git, and an internet connection for the initial dependency
installation. All reconstructed network inputs are bundled; no paper download,
MatACDC source program or local workspace directory is required to run.

```bash
git clone https://github.com/ninadsonawane/sequential_opf.git
cd sequential_opf
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell (alternative)
.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements-lock.txt
python reproduce.py --mode quick
```

Quick mode checks all three stages: IEEE solver accuracy, independent full
25-bus circuit validation, and feasible peak-hour B/C OPFs. It is **not** the
full timing or hourly study. For all planning points, hourly cases and benchmarks:

```bash
python reproduce.py --mode full
```

The full run can take several minutes or longer depending on CPU, JIT cache and
process startup. Raw results and reports are written under each stage's ignored
`results/` directory. `recorded_results/` contains the committed baseline artifacts
and is never overwritten by the runner. Read the stage README for individual
commands and timing repetitions.

## Recorded results

- [Stage 1 report](experiments/01_numba_benchmark/recorded_results/report.md):
  numerical correctness, equal-work compilation control and raw batch timings.
- [Stage 2 report](experiments/02_sequential_pf_25bus/recorded_results/report.html):
  full circuit, paper-output discrepancies, capability flags and conservation.
- [Stage 3 report](experiments/03_sequential_opf/recorded_results/report.html):
  24 equations, Table I/II analogues, hourly A/B/C figures, voltage/loss tradeoffs
  and complete-OPF timing comparison. Download/open HTML locally for the figures;
  GitHub does not render HTML as a website.

The initial recorded stage-3 study has 10 planning and 72 feasible operating
points under the declared limits. At peak load: A **29.27 MW**, B **26.58 MW**,
C **26.53 MW** losses. The selected LF rating is **345 kV** and Case C frequency
approximately **7.76 Hz**. Daily loss-energy reduction is **7.91% versus A**,
with only **0.19% incremental saving versus B**. These are scenario-specific
results, not utility measurements or general savings estimates.

## Data, assumptions and interpretation

- Stage 1 uses standard IEEE data distributed by installed PYPOWER and includes
  saved input/solution NPZ files. Its replicated AC batches are uncoupled.
- Stage 2 includes [every reconstructed numerical input](experiments/02_sequential_pf_25bus/data/reconstructed_25bus.json),
  including initial grids, bus/generator/branch tables, converter parameters
  and DC admittance. [Data provenance](experiments/02_sequential_pf_25bus/data/README.md)
  explains which values are assumed. The source programs and paper PDFs are
  not redistributed.
- Stage 3 adds a declared synthetic demand profile and equipment ratings,
  fully listed in `recorded_results/assumptions.json`. Generator voltage targets
  are controls here, extending P1's fixed targets. Planning retains line limits.
- All models are balanced positive sequence, **not unbalanced multiphase**.
  The OPF is a continuous local NLP, **not P1's mixed-integer PCPDIPM solver**.
  There are no controllable capacitor banks, switching penalties, ML inference,
  storage/ramp coupling or EMT/hardware certification.
- Convergence, steady-state constraint feasibility, independent electrical
  validation and global optimality are different claims. Only the first three
  are checked where stated. Stage 2 flags capability violations; Stage 3
  imposes a different, explicitly declared rating scenario.

## Verification and timing

Independent Newton checks verify the AC cases. A separate expanded-circuit
simultaneous solve checks the complete hybrid PF from a separate starting point.
Stage 3 independently checks all 10 planning states and six min/peak operating
states, plus conservation and every exported constraint. Raw timing samples,
medians/dispersion, environment details and tolerances accompany the results.

Warm benchmarks use persistent initialized pools and one native-library thread;
startup/JIT are excluded and first invocations recorded separately. Stage 2
compiles the AC kernel; Stage 3 compiles the **whole PF coordination core**.
Do not compare their speedup factors as the same workload. Hardware and floating
point arithmetic can change timing and Anderson iteration counts; bitwise
identity or matching laptop runtime is not a reproduction requirement.

See [methodology](docs/METHODOLOGY.md), [reproduction guide](docs/REPRODUCIBILITY.md)
and [third-party notices](THIRD_PARTY_NOTICES.md). Numerical inputs, CSV/JSON
outputs and plotting code are provided so claims can be checked independently.
