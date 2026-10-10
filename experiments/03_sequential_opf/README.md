# 3. P1-style planning and operation with sequential PF and SLSQP

**Question:** can a verified sequential PF act as the inner state evaluator for
constrained planning and loss-minimizing operation on the reconstructed case?

`opf_core.py` compiles the entire GS/Anderson + VSC/DC/B2B coordination core.
`optimization.py` defines the controls, physical constraints, loss accounting
and reduced-space continuous SLSQP optimizer. It also implements a separate
expanded-circuit reference from a separate starting point.

## Reproduce

```bash
python experiments/03_sequential_opf/run_study.py
python experiments/03_sequential_opf/run_tradeoff.py
python experiments/03_sequential_opf/benchmark_opf.py --opf-repeats 3 --pf-repeats 7
python experiments/03_sequential_opf/make_report.py
python experiments/03_sequential_opf/verify_results.py
```

The Stage 2 numerical JSON supplies the base network. Its committed
`recorded_results/solution.npz` is used only for a same-input legacy regression;
all optimization candidates are solved afresh. No paper or `_deps` folder is
needed. For a short numerical check: `python smoke.py` in this directory, or
`python reproduce.py --mode quick` from the root.

## Decisions and objectives

Planning minimizes `loss_MW + lambda * LF_rated_kV` at minimum/peak demand,
for lambda 0, 0.1, 0.2, 0.3, 0.4 MW/kV. Voltage range 69–500 kV;
frequency range 1–60 Hz. Fixed demand makes loss minimization equivalent to
generation minimization up to a constant. Lambda is a preference weight,
not a dollar capital-cost model. The next declared standard rating above
the lambda=0.1 peak continuous solution is selected and rechecked hourly.

| Case | Controls | Fixed quantities |
|---|---|---|
| A | Given feasible schedules, constructed by distance-based feasibility repair | Selected LF rating; 10 Hz |
| B | Generator/converter active/reactive dispatch and seven generator voltage targets | Selected LF rating; 10 Hz |
| C | B's controls plus LF frequency | Selected LF rating |

A's repair objective minimizes distance from a declared heuristic, not losses.
It is not a certified nearest projection; preparation effort is saved separately.
All cases keep LF generation at 200 MW and VSC1/3 active transfer obligations
at −200/+100 MW into AC. Dependent slack/voltage-controller outputs are checked
against declared bounds too; loss savings cannot come from dropping obligations.

Optional peak objective: `loss_MW + beta * sum_AC((abs(V)-1)**2)`, with beta
0/1000/5000 MW/p.u.². A constant-resistance sensitivity accompanies the main
disclosed piecewise R(f) model. No globally certified Pareto frontier is claimed.

## Differences from P1 and claim limits

- Continuous local SLSQP, not P1's custom mixed-integer PCPDIPM solver.
- Generator voltage targets are controls; P1 held them fixed.
- No controllable capacitor bank or capacitor-switching penalty.
- Operating line limits are retained in planning too.
- Ratings and the coincident 24-hour load profile are declared synthetic inputs.
- No unbalanced phases, ramps/storage coupling, ML inference, hardware V/f,
  investment-cost, EMT or protection validation.

Controls are scaled to a unit box; objective and constraint Jacobians share
central finite-difference PF evaluations. Inner AC mismatch tolerance `1e-10`
p.u.; interface tolerance `1e-9` p.u. A point is accepted only after optimizer
success, constraint/bound checks and power conservation. Independent circuit
solves check all 10 planning states and six min/peak A/B/C states; they do not
prove global optimality. Alternative frequency starts are checked at min/peak.

## Timing scope

Four backends: Python + two processes, Numba serial, Numba + two processes,
Numba + two threads. Two **independent complete peak OPFs** B/C are timed from
common fixed starts; a separate workload evaluates exactly 72 saved PF points.
JIT/pool startup is excluded from warm timings and first invocations saved
separately. Threads share immutable case data; each task owns its PF/optimizer
arrays. GS bus order and tolerances are preserved; no fastmath is used.

The recorded 9.54× two-OPF gain is our compiled full-PF implementation against
our Python-process implementation on one laptop. Stage 2 compiled only the AC
kernel, so its ratio describes a different scope. Neither result measures an
author-code improvement or establishes large-network scaling.

## Artifacts

[Recorded report](recorded_results/report.html) contains 24 equations and plots.
CSV/JSON files include planning Table I, paired converter Table II analogue,
all hourly P/Q/voltage states, every constraint margin, solver statuses,
independent checks, raw runtime repetitions, assumptions and source hashes.
`original_workspace_source_hashes.json` identifies the original measurement
code; new `results/source_hashes.json` identifies the portable rerun code.
