# Reproduction checklist

1. Clone the repository and create a Python 3.12 virtual environment.
2. Install `requirements-lock.txt` into that environment. No globally installed
   workspace libraries or bundled executable paths are referenced by the code.
3. Run `python reproduce.py --mode quick`. This is the integration/accuracy check.
4. Run `python reproduce.py --mode full` for all planning/hourly solutions and
   measured benchmark repetitions. Run stages individually with `--stage 1/2/3`.
5. Compare states, residuals, constraints and loss accounting before comparing
   timings. Review `verification.json`, `artifact_audit.json`, all constraint
   margins and optimizer status records. Do not expect identical CPU runtime.

## Input completeness

The full numerical hybrid-case JSON includes both initial areas, the final
network, converter settings and DC conductance. Stage 3's controls, bounds and
synthetic profile are explicitly defined in code and exported as assumptions.
The committed legacy solution NPZ is a regression target, not a state used to
initialize the independent circuit solve. IEEE cases are available from the
installed pinned PYPOWER dependency; Stage 1 also exports NPZ snapshots.

## Recorded versus new results

`recorded_results/` is the committed 9 October 2026 measurement baseline, with
portable paths substituted in readable reports. Original measurement-code
hashes identify the code that produced those results, not the subsequent port.
`results/` is ignored and contains new runs from this repository. New reports
write source hashes of the portable numerical modules. Re-running does not
overwrite the committed baseline. Small floating-point output differences are
allowed within the declared residual and independent-solution tolerances.

The Stage 2 baseline contains original public Stagg5/IEEE14 numerical snapshots.
The portable validation loads only their input arrays and independently
recomputes Newton references, alongside the complete reconstructed case.
The timing study retains all three scopes: unmodified public-case pair,
initial reconstructed HVAC pair and full reconstruction. No MatACDC source
program is redistributed or required.

## Practical limits

- Tested numerical dependency set: Python 3.12, NumPy 2.3.5, SciPy 1.18.1,
  Numba 0.68.0 / llvmlite 0.50.0, PYPOWER 5.1.21, Matplotlib 3.11.2.
- Native libraries, core counts, power mode and OS scheduling affect runtime.
- Process pools use spawn and `if __name__ == '__main__'`; run the scripts as
  programs, not unguarded notebook cells.
- The full frequency/voltage study is nonlinear and nonconvex. Solver failure
  raises an error; it does not become a fabricated feasible result.
- No author exact data/code, utility observations or OEM rating verification
  is implied by a successful reproduction.
