# Numerical data for the reconstructed hybrid network

`reconstructed_25bus.json` is the frozen numerical input used by both Stage 2
and Stage 3. It includes:

- `settings`: per-unit base, frequency/voltage bases, line lengths, load scale
  and DC pole assumption.
- `hvac` / `lfac`: bus, generator and branch arrays using MATPOWER column order.
- `dc_ybus`: DC conductance matrix on the stated DC base.
- `converter`: transformer/reactor/filter and loss coefficients.
- `initial_grids`: two independent pre-tie HVAC cases, including their arrays.

Arrays are numerical data. The downloaded MatACDC `.m` program, installed
PYPOWER library source and paper PDFs are not included. Topology/base-load
provenance: Stagg/El-Abiad five-bus benchmark as examined via KU Leuven MatACDC,
and IEEE14 through PYPOWER/MATPOWER. Cite these original benchmarks and P3;
see [third-party notices](../../../THIRD_PARTY_NOTICES.md).

**Declared reconstruction assumptions:** base demands multiplied by five;
50 km internal HVAC lines at 345 kV using extrapolated appendix R/L/C;
unity internal taps; no extra fixed shunts; 1000 km, 500 kV HVAC tie;
two 1000 km nominal-pi LF lines at 10 Hz / 500 kV; DC conductor/pole resistance
selection and 320 kV base. Original case-14 shunts/taps are not silently reused.
All selected values are stored explicitly; none were fitted to paper outputs.

Bus labels 1–19 denote HVAC, 20–22 LFAC and 23–25 DC in reported results.
LF array bus numbers are local 1–3. Converter AC attachment indices in code are
zero-based. Powers use a 100 MVA base; AC voltage bases are line-to-line kV.
MATPOWER bus type 1=PQ, 2=PV, 3=reference. Converter power is positive into AC;
positive converter DC withdrawal is out of its DC link.

Column order (zero-based array indexing in Python):

| Array | Columns in order |
|---|---|
| bus | BUS_I, BUS_TYPE, PD [MW], QD [MVAr], GS [MW at 1 p.u.], BS [MVAr at 1 p.u.], BUS_AREA, VM [p.u.], VA [deg], BASE_KV, ZONE, VMAX, VMIN |
| gen (first 10) | GEN_BUS, PG [MW], QG [MVAr], QMAX, QMIN, VG [p.u.], MBASE [MVA], GEN_STATUS, PMAX, PMIN; remaining entries use MATPOWER's extended schema |
| branch | F_BUS, T_BUS, BR_R [p.u.], BR_X [p.u.], BR_B [p.u. total charging], RATE_A/B/C [MVA], TAP, SHIFT [deg], BR_STATUS, ANGMIN, ANGMAX |

Zero branch ratings in the frozen PF input do not represent zero physical
capability. Stage 2 solves fixed controls and flags the separate converter
limits; Stage 3 supplies its own explicit synthetic line/generator ratings.

Stage 3's demand profile and generator/line/VSC bounds are additional synthetic
scenario inputs, listed in its assumptions JSON and `optimization.py`.
Numerical data reproduction does not imply actual utility/OEM equipment design.
