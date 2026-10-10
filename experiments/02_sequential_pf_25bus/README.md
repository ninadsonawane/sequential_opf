# 2. Sequential PF on the complete reconstructed 25-bus circuit

**Question:** does sequential AC/converter/DC coordination solve the full
selected hybrid circuit, and can an independent formulation verify it?

The topology follows P3 Fig. 6: Stagg-derived five-bus + IEEE14-derived HVAC
areas, three LFAC buses, three DC buses, seven VSCs, the AC 4–10 tie, two B2B
links and a three-terminal DC triangle. Missing modified electrical inputs
are disclosed rather than calibrated to reported outputs.

## Reproduce

```bash
python experiments/02_sequential_pf_25bus/run_validation.py
python experiments/02_sequential_pf_25bus/run_benchmark.py --repeats 21
python experiments/02_sequential_pf_25bus/make_report.py
```

All case numbers, including pre-tie initial grids, come from
[`data/reconstructed_25bus.json`](data/reconstructed_25bus.json). There is no
download step, third-party source program or hidden local dependency folder.
`model.py` loads fresh arrays; `grid_utils.py` supplies explicit nodal/current
assembly; `published.py` contains numerical comparison targets only.

## Method

1. Solve the two initial HVAC areas independently with the Stage 1 GS–Anderson
   kernel. Initialize the combined HVAC state and add the electrical AC tie.
2. Solve the coupled 19-bus HVAC and 3-bus LFAC networks; retain fixed reference
   phasors at HVAC buses 1/6 and LF bus 21.
3. Recover voltage-controller Q and LF balancing power. Compute VSC KCL/KVL,
   switching losses and transformer/reactor Joule losses.
4. Solve DC voltages with DC24 fixed, then update dependent converter active
   powers to satisfy the DC grid and the two B2B link balances. Repeat until
   interface mismatch converges; independently check inner AC P/Q mismatch.

Inner tolerance `1e-9` p.u.; outer `1e-8` p.u.; Anderson depth 8, coefficient
threshold 1000. Compiled and Python paths use the same Stage 1 kernel.
Only **initial independent HVAC solves** are pool-parallel in this experiment;
the coupled network and remaining converter/DC coordination stay serial.

## Validation versus feasibility

A separate expanded-circuit simultaneous solve has 75 real unknowns and its own
flat start. It checks phasors, converter powers, AC/DC/B2B balances and total
generation minus demand minus losses. Standard Stagg5 and IEEE14 cases are independently
checked by Newton–Raphson from bundled numeric snapshots. These checks validate equations, not author input
provenance or global optimality.

This fixed-schedule PF **flags converter capability violations** under the
declared 100 MVA interpretation; it does not silently change schedules or
switch control modes. LFAC outputs closely match published values, while some
HVAC/DC outputs differ. The authors' exact modified case remains unavailable.

## Outputs

`results/solution.npz`, `verification.json`, `power_balance.json`, complete
`reconstructed_inputs.json`, paper comparison CSVs, converter limit flags,
coordination traces, notation and DC-pole sensitivities, timing samples and
report. The committed [recorded report](recorded_results/report.html) includes
the original public-source checks performed in the local study. The portable
runner reads their numerical bus/gen/branch snapshots, recomputes the independent
Newton references, and retains the same three benchmark workloads without
redistributing or requiring the downloaded source programs.

**Next:** [planning and operation with SLSQP](../03_sequential_opf/README.md).
