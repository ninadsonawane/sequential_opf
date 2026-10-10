# Methodology and claims

## Common physical conventions

Balanced positive-sequence AC phasors, 100 MVA base; AC nominal voltage is
line-to-line kV. `S = V * conj(Y @ V)`. Positive converter `Ps/Qs` is injected
into AC. Positive station DC power is withdrawn from its DC grid/link.
Slack P/Q and PV Q are dependent outputs; specifying a voltage target does not
specify Q. Distinguish rated kV, AC per-unit magnitude and DC-link voltage.

For the station circuit: `Is=conj(Ss/Vs)`, `Vf=Vs+Zt*Is`,
`Ic=Is+j*Bf*Vf`, `Vc=Vf+Zc*Ic`. DC withdrawal is
`Re(Vc*conj(Ic)) + switching_loss`. Transformer/reactor Joule loss is already
present in `Re(Sc)-Ps`; it must not be counted twice. Capacitive filter KCL is
an explicit interpretation of P3's ambiguous equation/table notation.

## Stage 1: algorithm versus execution

In-place Gauss–Seidel uses newly updated earlier buses, not independent bus
tasks. Anderson mixes residual/history differences and restores PV magnitudes.
The Python function and its Numba compilation are the same mathematical method,
but floating-point least squares can change iterates and iteration counts.
Verify final P/Q mismatch and independent Newton agreement. The fixed-100-sweep
plain-GS control isolates compilation from convergence-history differences.

## Stage 2: sequential interfaces

An AC tie electrically couples the two same-frequency areas after their initial
independent solves. They cannot remain independent problems during subsequent
coupled HVAC solves. DC nodal power balances and B2B losses determine dependent
converter active powers. Repeat the AC → station → DC/balance update sequence.
Check AC mismatch and interface residuals separately. Independent circuit
validation checks the model being solved; it cannot supply missing paper inputs.

## Stage 3: reduced-space optimization

The inner solver returns `s(u)` satisfying physical state equations. The outer
optimizer minimizes `J(s(u),u)` with state/control inequalities. It is not an
OPF merely because the PF converges; controls must be chosen and constraints
must be checked. Here SLSQP uses scaled controls and shared bounded central
finite differences, becoming one-sided at a bound.

Planning: `L + lambda*V_rated`, where fixed demand makes loss minimization
equivalent to total-generation minimization. These weights are not CAPEX prices.
Voltage rating changes the physical bases; frequency changes line/reactor X
and charging B. Operation holds the selected rating and compares feasible
given schedules, fixed-frequency optimization and free-frequency optimization.

The optional voltage-deviation objective is a separate extension, not P1's
capacitor-switching term. Minimizing losses alone can push voltage toward an
upper bound. Always inspect voltage and reactive dispatch, not only the scalar
loss objective. Local optimizer success plus alternative starts is not proof
of a globally optimal planning frequency or complete Pareto frontier.

## Timing boundaries

Stage 1: independent AC batches. Stage 2: initial independent HVAC pool stage
plus otherwise serial full coordination. Stage 3: complete compiled PF core,
with pools across independent OPFs/scenarios and SLSQP outside the compiled core.
Each benchmark saves raw samples, protocol and numerical comparisons.
Warm medians exclude startup/JIT. First workloads are not pure startup tests.
Do not compare incompatible scopes, claim author-code acceleration or infer
large-grid scaling from one reconstructed small circuit.
