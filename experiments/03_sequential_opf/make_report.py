"""Offline, source-traceable scientific report; regenerate from saved results."""

import opf_bootstrap
from pathlib import Path
import csv
import html
import json
import hashlib
import platform
import sys
from datetime import datetime, timezone, timedelta
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
import numba
import llvmlite
from formulation import build_formulation

HERE = opf_bootstrap.HERE
OUT = HERE / "results"
FIG = OUT / "figures"
COLORS = {"A": "#6f7886", "B": "#147c91", "C": "#c65a32"}
METHODS = ["python_process", "numba_serial", "numba_process", "numba_thread"]
METHOD_LABELS = [
    "Python + processes",
    "Numba serial",
    "Numba + processes",
    "Numba + threads",
]


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_csv(name):
    with (OUT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def num(row, field):
    return float(row[field])


def fmt(x, decimals=3):
    return f"{float(x):,.{decimals}f}"


def table(headers, rows, classes=""):
    return (
        '<div class="table-wrap"><table class="'
        + classes
        + '"><thead><tr>'
        + "".join("<th>" + html.escape(str(x)) + "</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + html.escape(str(x)) + "</td>" for x in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def save_plot(fig, name):
    fig.savefig(FIG / (name + ".svg"), bbox_inches="tight")
    fig.savefig(FIG / (name + ".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tag(name, caption):
    return f'<figure><img src="figures/{name}.svg" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def figures(hourly, planning, benchmark, tradeoff, summary):
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#253349",
            "text.color": "#253349",
            "axes.edgecolor": "#9aa5b2",
            "grid.color": "#e1e6ec",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
        }
    )
    groups = {
        k: sorted([r for r in hourly if r["case"] == k], key=lambda r: num(r, "hour"))
        for k in COLORS
    }
    hours = np.arange(24)
    fig, axes = plt.subplots(
        2, 1, figsize=(10.2, 6.7), sharex=True, layout="constrained"
    )
    for k, rows in groups.items():
        axes[0].plot(
            hours,
            [num(r, "total_loss_mw") for r in rows],
            label=f"Case {k}",
            color=COLORS[k],
            linewidth=2,
        )
        axes[1].plot(
            hours, [num(r, "loss_pct") for r in rows], color=COLORS[k], linewidth=2
        )
    axes[0].set(
        title="Total system losses across the synthetic day", ylabel="Loss (MW)"
    )
    axes[0].legend(ncol=3, loc="upper left")
    axes[1].set(ylabel="Loss / demand (%)", xlabel="Hour (0–23)")
    for ax in axes:
        ax.grid(alpha=0.7)
        ax.set_xticks(np.arange(0, 24, 2))
        ax.axvline(17, color="#253349", linestyle=":", alpha=0.5)
    save_plot(fig, "hourly_losses")

    fig, axes = plt.subplots(
        2, 1, figsize=(10.2, 5.8), sharex=True, layout="constrained"
    )
    axes[0].step(
        hours,
        [num(r, "demand_mw") for r in groups["A"]],
        where="mid",
        color="#253349",
        linewidth=2,
    )
    axes[0].set(
        title="Declared demand profile and optimal operating frequency",
        ylabel="Demand (MW)",
    )
    axes[1].plot(
        hours,
        [num(r, "frequency_hz") for r in groups["C"]],
        color=COLORS["C"],
        label="C: variable frequency",
        linewidth=2,
    )
    axes[1].axhline(10, color=COLORS["B"], linestyle="--", label="A / B: 10 Hz")
    axes[1].set(ylabel="LF frequency (Hz)", xlabel="Hour (0–23)", ylim=(7.4, 10.4))
    axes[1].legend(loc="upper left", ncol=2)
    for ax in axes:
        ax.grid(alpha=0.7)
        ax.set_xticks(np.arange(0, 24, 2))
    save_plot(fig, "demand_frequency")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), layout="constrained")
    for k, rows in groups.items():
        for ax, field in zip(axes, ("min_voltage_pu", "max_voltage_pu")):
            ax.plot(
                hours,
                [num(r, field) for r in rows],
                color=COLORS[k],
                label=k,
                linewidth=1.8,
            )
    axes[0].set(
        title="Minimum AC bus voltage", ylabel="Voltage (p.u.)", ylim=(0.93, 1.065)
    )
    axes[1].set(title="Maximum AC bus voltage", ylim=(0.93, 1.065))
    for ax in axes:
        ax.axhspan(0.94, 1.06, color="#147c91", alpha=0.045)
        for bound in (0.94, 1.06):
            ax.axhline(bound, color="#9aa5b2", linestyle=":", linewidth=1)
        ax.set(xlabel="Hour (0–23)", xticks=np.arange(0, 24, 4))
        ax.grid(alpha=0.5)
    axes[0].legend(title="Case", ncol=3)
    save_plot(fig, "voltage_envelope")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), layout="constrained")
    for period, color, style in [
        ("minimum", "#147c91", "-"),
        ("peak", "#c65a32", "--"),
    ]:
        pp = [r for r in planning if r["load_period"] == period]
        xx = [num(r, "alpha2_mw_per_kv") for r in pp]
        for ax, field in zip(axes, ("voltage_rating_kv", "frequency_hz")):
            ax.plot(
                xx,
                [num(r, field) for r in pp],
                color=color,
                linestyle=style,
                marker="o",
                label=period.capitalize(),
            )
    axes[0].set(title="Continuous planning voltage", ylabel="Rated LF voltage (kV)")
    axes[1].set(title="Frequency co-optimized with voltage", ylabel="LF frequency (Hz)")
    for ax in axes:
        ax.set(xlabel="Voltage penalty λ (MW/kV)")
        ax.grid(alpha=0.7)
        ax.legend()
    save_plot(fig, "planning_weights")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), layout="constrained")
    pp = [r for r in planning if r["load_period"] == "peak"]
    axes[0].plot(
        [num(r, "voltage_rating_kv") for r in pp],
        [num(r, "total_loss_mw") for r in pp],
        "o-",
        color="#147c91",
    )
    for r in pp:
        axes[0].annotate(
            "λ=" + fmt(num(r, "alpha2_mw_per_kv"), 1),
            (num(r, "voltage_rating_kv"), num(r, "total_loss_mw")),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[0].set(
        title="Planning: loss–voltage tradeoff",
        xlabel="Rated LF voltage (kV)",
        ylabel="Peak loss (MW)",
        xlim=(245, 555),
        ylim=(23, 32.2),
    )
    tx = [num(r, "voltage_deviation_pu_squared") for r in tradeoff]
    ty = [num(r, "total_loss_mw") for r in tradeoff]
    axes[1].plot(tx, ty, "o-", color=COLORS["C"])
    for r in tradeoff:
        axes[1].annotate(
            "β=" + fmt(num(r, "beta_mw_per_pu_squared"), 0),
            (num(r, "voltage_deviation_pu_squared"), num(r, "total_loss_mw")),
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=8,
            ha="center",
        )
    axes[1].set(
        title="Operation: loss–voltage deviation tradeoff",
        xlabel="Σ (|V| − 1)² (p.u.²)",
        ylabel="Peak loss (MW)",
        ylim=(26.4, 27.85),
        xlim=(0.0006, 0.0117),
    )
    for ax in axes:
        ax.grid(alpha=0.6)
    save_plot(fig, "multiobjective_tradeoffs")

    fig, ax = plt.subplots(figsize=(10.2, 3.8), layout="constrained")
    bottom = np.zeros(3)
    components = [
        ("hvac_line_loss_mw", "HVAC lines", "#718799"),
        ("lfac_line_loss_mw", "LFAC lines", "#80bec6"),
        ("dc_line_loss_mw", "DC lines", "#ead69a"),
        ("converter_loss_mw", "Converters", "#c65a32"),
    ]
    for field, label, color in components:
        values = np.array([summary["peak"][k][field] for k in COLORS])
        ax.bar(np.arange(3), values, bottom=bottom, label=label, color=color, width=0.5)
        bottom += values
    for i, v in enumerate(bottom):
        ax.text(i, v + 0.3, fmt(v, 2) + " MW", ha="center", fontweight="bold")
    ax.set(
        title="Peak-hour loss breakdown",
        ylabel="Loss (MW)",
        xticks=np.arange(3),
        xticklabels=[
            "A: feasible PF",
            "B: fixed-frequency OPF",
            "C: variable-frequency OPF",
        ],
        ylim=(0, 34),
    )
    ax.legend(ncol=4, loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    save_plot(fig, "loss_breakdown")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), layout="constrained")
    for ax, scope, title in zip(
        axes,
        ("complete_opf_peak_B_C", "fixed_work_72_solved_PF_points"),
        ("Two complete peak-hour OPFs (B + C)", "72 fixed-work PF evaluations"),
    ):
        rr = [
            next(r for r in benchmark if r["scope"] == scope and r["method"] == m)
            for m in METHODS
        ]
        med = np.array([num(r, "median_seconds") for r in rr])
        lo = np.array([num(r, "min_seconds") for r in rr])
        hi = np.array([num(r, "max_seconds") for r in rr])
        ax.barh(
            np.arange(4),
            med,
            color=["#718799", "#b4cfd2", "#448d9a", "#c65a32"],
            height=0.55,
        )
        ax.errorbar(
            med,
            np.arange(4),
            xerr=np.vstack([med - lo, hi - med]),
            fmt="none",
            ecolor="#253349",
            capsize=3,
        )
        for i, r in enumerate(rr):
            ax.text(
                hi[i] + med.max() * 0.025,
                i,
                fmt(med[i], 3) + " s",
                va="center",
                fontsize=9,
            )
        ax.set(
            title=title,
            xlabel="Warm elapsed time (seconds)",
            yticks=np.arange(4),
            yticklabels=METHOD_LABELS,
            xlim=(0, hi.max() * 1.25),
        )
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.5)
        ax.set_axisbelow(True)
    save_plot(fig, "runtime_comparison")

    fig, ax = plt.subplots(figsize=(10.2, 3.6), layout="constrained")
    for k in ("B", "C"):
        ax.plot(
            hours,
            [num(r, "solve_wall_seconds") for r in groups[k]],
            color=COLORS[k],
            marker=".",
            label=k,
        )
    ax.set(
        title="Recorded warm optimizer time by hour (Numba PF, serial study)",
        ylabel="One OPF solve (seconds)",
        xlabel="Hour (0–23)",
        xticks=np.arange(0, 24, 2),
    )
    ax.legend(title="Case", ncol=2)
    ax.grid(alpha=0.6)
    save_plot(fig, "hourly_optimizer_time")


def main():
    summary = read_json("summary.json")
    assumptions = read_json("assumptions.json")
    hourly = read_csv("hourly_results.csv")
    planning = read_csv("table1_planning.csv")
    benchmark = read_csv("benchmark_summary.csv")
    tradeoff = read_csv("operational_multiobjective_tradeoff.csv")
    dispatch = read_csv("table2_converter_dispatch_peak.csv")
    gens = read_csv("table2_generator_dispatch_peak.csv")
    volts = read_csv("bus_voltages_all_hours.csv")
    design = read_json("design_selection.json")
    sensitivity = read_json("resistance_model_sensitivity.json")
    figures(hourly, planning, benchmark, tradeoff, summary)
    equations = build_formulation()

    peak_rows = []
    for k, label in [
        ("A", "Feasible fixed schedules / PF"),
        ("B", "Dispatch + generator voltage OPF"),
        ("C", "B + frequency optimization"),
    ]:
        p = summary["peak"][k]
        peak_rows.append(
            [
                k,
                label,
                fmt(p["frequency_hz"]),
                fmt(p["total_loss_mw"]),
                fmt(p["loss_pct"]),
                fmt(summary["daily_loss_energy_mwh"][k]),
                fmt(p["voltage_deviation"], 6),
            ]
        )
    peak_table = table(
        [
            "Case",
            "Decision scope",
            "Frequency (Hz)",
            "Peak loss (MW)",
            "Loss / demand (%)",
            "24 h loss (MWh)",
            "Peak ΣΔV² (p.u.²)",
        ],
        peak_rows,
    )

    plan_rows = []
    for weight in (0, 0.1, 0.2, 0.3, 0.4):
        pair = [
            next(
                r
                for r in planning
                if r["load_period"] == p and num(r, "alpha2_mw_per_kv") == weight
            )
            for p in ("minimum", "peak")
        ]
        plan_rows.append(
            [fmt(weight, 1)]
            + [
                fmt(num(r, f))
                for r in pair
                for f in (
                    "voltage_rating_kv",
                    "frequency_hz",
                    "lf_grid_loss_mw",
                    "total_loss_mw",
                )
            ]
        )
    plan_table = table(
        [
            "λ (MW/kV)",
            "Min: kV",
            "Min: Hz",
            "Min: LF loss (MW)",
            "Min: total (MW)",
            "Peak: kV",
            "Peak: Hz",
            "Peak: LF loss (MW)",
            "Peak: total (MW)",
        ],
        plan_rows,
    )

    pair_rows = []
    pair_data = []
    for label, hv, lf in [("B2B A", 4, 6), ("B2B B", 5, 7)]:
        d = [
            next(r for r in dispatch if r["case"] == "C" and r["station"] == f"VSC{i}")
            for i in (hv, lf)
        ]
        numeric = {
            f"{side}_{name}": num(r, field)
            for side, r in zip(("HVAC", "LFAC"), d)
            for name, field in [
                ("P_mw", "p_ac_mw"),
                ("Q_mvar", "q_ac_mvar"),
                ("V_pu", "terminal_voltage_pu"),
            ]
        }
        pair_data.append(
            dict(
                pair=label, HVAC_station=f"VSC{hv}", LFAC_station=f"VSC{lf}", **numeric
            )
        )
        pair_rows.append(
            [f"{label} (VSC{hv} / VSC{lf})"]
            + [
                fmt(numeric[f"{side}_{name}"])
                for side in ("HVAC", "LFAC")
                for name in ("P_mw", "Q_mvar", "V_pu")
            ]
        )
    with (OUT / "table2_b2b_caseC_peak.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=list(pair_data[0]))
        w.writeheader()
        w.writerows(pair_data)
    pair_table = table(
        [
            "B2B pair",
            "HVAC P (MW)",
            "HVAC Q (MVAr)",
            "HVAC V (p.u.)",
            "LFAC P (MW)",
            "LFAC Q (MVAr)",
            "LFAC V (p.u.)",
        ],
        pair_rows,
    )
    station_table = table(
        [
            "Case",
            "Station",
            "PCC bus",
            "P into AC (MW)",
            "Q into AC (MVAr)",
            "PCC V (p.u.)",
            "Converter V (p.u.)",
            "Station loss (MW)",
        ],
        [
            [r["case"], r["station"], r["pcc_bus"]]
            + [
                fmt(num(r, k))
                for k in (
                    "p_ac_mw",
                    "q_ac_mvar",
                    "terminal_voltage_pu",
                    "converter_voltage_pu",
                    "station_loss_mw",
                )
            ]
            for r in dispatch
        ],
    )
    gen_rows = []
    for bus in assumptions["generator_buses"]:
        gg = [
            next(r for r in gens if int(r["bus"]) == bus and r["case"] == k)
            for k in COLORS
        ]
        row = [bus]
        for r in gg:
            v = next(
                v
                for v in volts
                if int(v["hour"]) == 17
                and int(v["bus"]) == bus
                and v["case"] == r["case"]
            )
            row.extend(
                [
                    fmt(num(r, "p_generation_mw")),
                    fmt(num(r, "q_generation_mvar")),
                    fmt(num(v, "voltage_pu")),
                ]
            )
        gen_rows.append(row)
    generator_table = table(
        [
            "Bus",
            "A: P (MW)",
            "A: Q (MVAr)",
            "A: V (p.u.)",
            "B: P (MW)",
            "B: Q (MVAr)",
            "B: V (p.u.)",
            "C: P (MW)",
            "C: Q (MVAr)",
            "C: V (p.u.)",
        ],
        gen_rows,
    )
    runtime_table = table(
        [
            "Workload",
            "Backend",
            "Repeats",
            "Median (s)",
            "Range (s)",
            "Speedup vs our Python",
        ],
        [
            [
                (
                    "Two complete OPFs"
                    if r["scope"].startswith("complete")
                    else "72 fixed PFs"
                ),
                METHOD_LABELS[METHODS.index(r["method"])],
                r["repeats"],
                fmt(num(r, "median_seconds")),
                fmt(num(r, "min_seconds")) + "–" + fmt(num(r, "max_seconds")),
                fmt(num(r, "ratio_vs_our_python_process"), 2) + "×",
            ]
            for r in benchmark
        ],
    )
    tradeoff_table = table(
        [
            "β (MW/p.u.²)",
            "Peak loss (MW)",
            "ΣΔV² (p.u.²)",
            "Frequency (Hz)",
            "Max V (p.u.)",
        ],
        [
            [
                fmt(num(r, "beta_mw_per_pu_squared"), 0),
                fmt(num(r, "total_loss_mw")),
                fmt(num(r, "voltage_deviation_pu_squared"), 6),
                fmt(num(r, "frequency_hz")),
                fmt(num(r, "max_voltage_pu")),
            ]
            for r in tradeoff
        ],
    )
    provenance = table(
        ["Component", "What is used", "Status"],
        [
            [
                "Network",
                "P3 Fig. 6: 19 HVAC, 3 LFAC, 3 DC buses; 7 VSCs",
                "Published topology; reconstructed electrical inputs",
            ],
            [
                "HVAC electrical data",
                "Public Stagg 5 / IEEE 14 topology; 50 km internal lines at 345 kV; P3 appendix RLC extrapolated to internal geometry",
                "Declared reconstruction assumptions",
            ],
            [
                "Loads",
                "Public loads multiplied by 5, then the declared hourly factor; peak 2120 MW",
                "Not the authors’ modified case or utility measurements",
            ],
            [
                "Long lines / DC",
                "Two 1000 km LF lines, 1000 km HVAC tie; nominal-pi AC; assumed two-conductor DC resistance",
                "P3-derived parameters plus disclosed assumptions",
            ],
            [
                "Equipment ratings / forecast",
                "Synthetic generator and line bounds, 200 MVA reference converter family, coincident 24 h profile",
                "Benchmark scenario, not OEM or author data",
            ],
            [
                "PF algorithm",
                "Sequential AC/converter/DC coordination + GS/Anderson; retained reference angles at HVAC buses 1/6 and LF bus 21",
                "Our implementation; not the authors’ source code",
            ],
            [
                "OPF",
                "P1-style loss/voltage objectives, continuous controls, SciPy SLSQP",
                "Adaptation, not the P1 custom mixed-integer solver",
            ],
        ],
    )
    controls_table = table(
        ["Quantity", "Decision / dependent role"],
        [
            [
                "Planning",
                "20 controls: five HVAC Pg, VSC5 P, five VSC Q, LF frequency, LF rated kV, seven generator Vm",
            ],
            [
                "Operation B",
                "18 controls: planning controls with rated kV and frequency fixed",
            ],
            ["Operation C", "19 controls: planning controls with rated kV fixed"],
            [
                "Fixed obligations",
                "LF generator bus 20: 200 MW; VSC1: −200 MW; VSC3: +100 MW",
            ],
            [
                "Voltage setpoints",
                "VSC2 PCC bus 17: 0.985 p.u.; LF reference bus 21: 1 p.u.; DC reference bus 24: 1 p.u.",
            ],
            [
                "Dependent outputs",
                "PQ bus V/θ, PV bus Q, slack P/Q, VSC2/4/7 P, VSC2/6 Q, DC voltages and losses",
            ],
        ],
    )
    ratings_table = table(
        ["Constraint", "Declared bounds / assumption"],
        [
            ["AC / DC bus voltage", "0.94–1.06 p.u. / 0.95–1.05 p.u."],
            [
                "Generator controlled Vm",
                "0.96–1.04 p.u.; added as an explicit extension of P1’s fixed Vsch",
            ],
            [
                "Generator Pmax (buses 1,2,6,7,8,11,13,20)",
                "1250, 1500, 1662, 700, 500, 500, 500, 250 MW",
            ],
            [
                "Generator Q / S",
                "HVAC Qmax = 0.75 Pmax; LF Qmax = 400 MVAr; Smax = √(Pmax²+Qmax²)",
            ],
            [
                "Converter current / voltage",
                "Iref ≤ 2.1 p.u. on 100 MVA reference base; converter voltage 0.85–1.20 p.u.",
            ],
            ["Line current (both ends)", "HVAC 1500 A, LFAC 1000 A, DC 1000 A"],
            ["Line active power (both ends)", "HVAC 1000 MW, LFAC 600 MW"],
            [
                "Planning ranges",
                "69–500 kV, 1–60 Hz; same operating limits also retained in planning",
            ],
            [
                "Converter Q controls / VSC5 P",
                "±150 MVAr / 0–200 MW; dependent transfer directions constrained",
            ],
        ],
    )
    checked = len(
        [
            r
            for r in read_csv("constraint_audit_all_hours.csv")
            if r["satisfied"] == "True"
        ]
    )
    checks_table = table(
        ["Check", "Evidence"],
        [
            [
                "Feasible operating points",
                "72 / 72, checked against all declared input and state constraints",
            ],
            [
                "Planning solutions",
                "10 / 10, SLSQP successful and checked against declared constraints",
            ],
            [
                "Constraint records",
                f"{checked:,} satisfied records; numerical acceptance margin −1e−6",
            ],
            [
                "Power conservation",
                f'Max |ΣPg − Pd − L| = {summary["max_conservation_error_mw"]:.3e} MW',
            ],
            [
                "Independent PF validation",
                "Expanded-circuit simultaneous solve, separate flat start: all 10 planning and 6 min/peak operating points",
            ],
            [
                "Independent voltage agreement",
                f'Max operating-point difference = {summary["max_independent_voltage_error_pu"]:.3e} p.u.',
            ],
            [
                "Legacy case regression",
                "Same inputs at 500 kV / 10 Hz reproduce the previous reconstruction within 1e−7 p.u.",
            ],
            [
                "Backend comparison",
                "All 40 timed batches passed objective / phasor comparisons; source and tolerances held fixed",
            ],
            [
                "Alternative local starts",
                "5 Hz / 20 Hz starts at min and peak: final loss agreement within 2e−7 MW; no global certificate",
            ],
        ],
    )
    reference_links = """<p><b>P1:</b> Nguyen &amp; Santoso, <i>Optimal Planning and Operation of Multi-Frequency HVac Transmission Systems</i>, IEEE TPWRS 36(1), 2021. <a href="https://doi.org/10.1109/TPWRS.2020.3037967">DOI / publisher</a>; <a href="https://doi.org/10.1109/TPWRS.2020.3037967">publisher source (PDF not redistributed)</a>.</p>
    <p><b>P3:</b> Kim, Nguyen &amp; Santoso, <i>A sequential power flow algorithm…</i>, 2026. <a href="https://doi.org/10.1016/j.ijepes.2026.112025">DOI / publisher</a>; <a href="https://doi.org/10.1016/j.ijepes.2026.112025">publisher source (PDF not redistributed)</a>. See the <a href="../../02_sequential_pf_25bus/recorded_results/report.html">previous reconstruction audit</a> for missing inputs and converter notation.</p>"""
    assets = [
        ("table1_planning.csv", "Planning Table I data"),
        ("table2_b2b_caseC_peak.csv", "B2B Table II analogue"),
        ("table2_converter_dispatch_peak.csv", "All peak converter values"),
        ("table2_generator_dispatch_peak.csv", "All peak generator values"),
        ("hourly_results.csv", "24 h A/B/C results"),
        ("converter_dispatch_all_hours.csv", "All hourly converter dispatch"),
        ("generator_dispatch_all_hours.csv", "All hourly generator dispatch"),
        ("bus_voltages_all_hours.csv", "All bus voltages"),
        ("constraint_audit_all_hours.csv", "Every operating constraint"),
        ("baseline_feasibility_preparation.csv", "Baseline preparation time"),
        ("benchmark_summary.csv", "Runtime table"),
        ("benchmark_samples.csv", "Every timing sample"),
        ("benchmark_first_calls.csv", "First runner invocations"),
        ("benchmark_protocol.json", "Benchmark protocol"),
        ("operational_multiobjective_tradeoff.csv", "Operational objective tradeoff"),
        ("resistance_model_sensitivity.json", "R(F) sensitivity"),
        ("independent_validation.json", "Independent operating verification"),
        ("planning_details.json", "Planning controls and verification"),
        ("solution_controls.json", "All saved control vectors"),
        ("optimizer_records.json", "Solver status for every point"),
        ("assumptions.json", "Complete assumptions"),
        ("formulation.txt", "Equations as text"),
        ("source_hashes.json", "Numerical source hashes"),
    ]
    links = (
        '<ul class="assets">'
        + "".join(f'<li><a href="{name}">{label}</a></li>' for name, label in assets)
        + "</ul>"
    )
    speed = next(
        num(r, "ratio_vs_our_python_process")
        for r in benchmark
        if r["scope"].startswith("complete") and r["method"] == "numba_thread"
    )
    warm_seconds = {
        r["method"]: num(r, "median_seconds")
        for r in benchmark
        if r["scope"].startswith("complete")
    }
    generated_date = (
        datetime.now(timezone(timedelta(hours=5, minutes=30)))
        .strftime("%d %B %Y")
        .lstrip("0")
    )
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>25-bus sequential PF — P1-style planning and operation</title>
    <style>
    *{{box-sizing:border-box}}body{{margin:0;background:#edf1f5;color:#243248;font:16px/1.6 'Segoe UI',Arial,sans-serif}}main{{max-width:1160px;margin:24px auto;background:white;padding:42px 44px;border-radius:12px}}h1{{font-size:34px;line-height:1.2;letter-spacing:-.5px;margin:12px 0 18px}}h2{{font-size:25px;line-height:1.3;margin:44px 0 15px}}h3{{font-size:18px;margin:22px 0 10px}}p{{margin:12px 0}}a{{color:#147c91}}.eyebrow{{text-transform:uppercase;letter-spacing:1.6px;font-size:12px;font-weight:700;color:#147c91}}.subtitle{{font-size:19px;line-height:1.5;color:#536175}}.scope{{background:#eff5f7;border-left:4px solid #147c91;padding:16px 20px;margin:22px 0}}.scope p{{margin:4px 0}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.card{{padding:17px;background:#f5f7fa;border-radius:8px}}.card b{{display:block;font-size:26px;color:#142d43}}.card span{{font-size:13px;color:#59697b}}nav{{display:flex;flex-wrap:wrap;gap:10px 22px;padding:14px 0;border-top:1px solid #e2e8ee;border-bottom:1px solid #e2e8ee;font-size:14px}}.table-wrap{{overflow-x:auto;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:13px;line-height:1.4}}th{{background:#173247;color:white;text-align:right;font-weight:600;padding:12px 10px}}th:first-child,td:first-child{{text-align:left}}td{{text-align:right;padding:10px;border-bottom:1px solid #e0e7ee}}tr:nth-child(even) td{{background:#f4f7fa}}td:nth-child(2){{text-align:left}}figure{{margin:24px 0}}figure img{{width:100%;height:auto;display:block}}figcaption{{font-size:13px;color:#59697b;padding:8px 3px}}.equation{{border-bottom:1px solid #e4e9ee;padding:10px 0}}.equation img{{max-width:100%;height:auto;display:block;max-height:94px}}.equation p{{font-size:14px}}details{{border:1px solid #d7e0e9;border-radius:7px;padding:14px 18px;margin:18px 0}}summary{{cursor:pointer;font-weight:600}}.flow{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:22px 0}}.step{{flex:1;min-width:180px;padding:16px;background:#f1f5f8;border-radius:6px;line-height:1.45;font-size:14px}}.step b{{display:block;color:#147c91;margin-bottom:6px}}.small{{font-size:13px;color:#59697b}}.assets{{columns:2;font-size:14px;padding-left:22px}}.callout{{padding:15px 20px;border-left:4px solid #c65a32;background:#fcf4ef}}code{{font-family:Consolas,monospace;font-size:13px;background:#f1f4f7;padding:2px 5px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f7;padding:18px;border-radius:6px;font-size:13px}}footer{{margin-top:36px;border-top:1px solid #e0e7ee;padding-top:18px;color:#59697b;font-size:13px}}@media(max-width:760px){{main{{margin:0;padding:24px 18px;border-radius:0}}h1{{font-size:28px}}.cards{{grid-template-columns:repeat(2,1fr)}}.assets{{columns:1}}}}@media print{{body{{background:white}}main{{margin:0;padding:10px;max-width:none}}nav{{display:none}}details{{break-inside:avoid}}figure{{break-inside:avoid}}}}
    </style></head><body><main>
    <div class="eyebrow">Research experiment · Report generated {generated_date} (IST) · Ninad / Seiole</div>
    <h1>Optimal planning and operation<br>with sequential Gauss–Seidel + Anderson PF</h1>
    <p class="subtitle">P1-style objectives on the reconstructed P3 25-bus HVAC / LFAC / MTDC network, with a compiled PF solver and independently checked electrical states.</p>
    <div class="scope"><p><b>Completed:</b> 10 planning points, 72 hourly A/B/C points, dispatch tables, multi-objective tradeoffs and full-OPF timing comparisons.</p><p><b>Scope:</b> an adapted, balanced positive-sequence benchmark using disclosed ratings and a synthetic day. This is not the authors’ exact data/code reproduction, an unbalanced multiphase model, or an equipment-design certification. Solutions are local NLP results.</p></div>
    <div class="cards"><div class="card"><b>345 kV</b><span>Selected LF design rating</span></div><div class="card"><b>7.76 Hz</b><span>Case C peak operating frequency</span></div><div class="card"><b>{summary['C_energy_reduction_pct']:.2f}%</b><span>Daily loss-energy reduction, C vs A</span></div><div class="card"><b>{speed:.2f}×</b><span>Warm two-OPF batch speedup, our backends</span></div></div>
    <nav><a href="#method">Method</a><a href="#planning">Planning / Table I</a><a href="#operation">Cases A/B/C</a><a href="#dispatch">Dispatch / Table II</a><a href="#runtime">Runtime</a><a href="#equations">Equations</a><a href="#audit">Assumptions / verification</a><a href="#files">Data / reproduction</a></nav>

    <h2 id="method">1. What is being optimized, and why PF sits inside OPF</h2>
    <p>Power flow answers: <i>given controls, what electrical state follows?</i> The optimizer answers: <i>which feasible controls reduce the objective?</i> We use P1’s planning/operation distinction and P3-inspired sequential PF to evaluate each candidate. Gauss–Seidel and Anderson solve the state equations; SLSQP chooses the controls. Numba speeds up those numerical calculations and does not change the objective.</p>
    <div class="flow"><div class="step"><b>Outer optimizer</b>Choose generator/converter settings; planning also chooses rated kV and frequency.</div><span>→</span><div class="step"><b>Inner sequential PF</b>AC GS + Anderson → converter losses → DC/B2B balances → update dependent injections; repeat.</div><span>→</span><div class="step"><b>Evaluate candidate</b>Compute total loss, physical line currents, voltages, generator/VSC limits; return objective and constraints.</div></div>
    {controls_table}
    <p>Planning trades lower operating losses against lower voltage rating using <b>L + λV<sub>rated</sub></b>. Fixed demand makes this equivalent to P1’s total-generation-plus-voltage objective up to a constant. The λ weights are numerical preference weights, not estimates of investment dollars. Operation fixes the chosen voltage rating and minimizes loss. A separate experiment adds voltage deviation as a second operational objective.</p>

    <h2 id="planning">2. Planning — Table I analogue</h2>
    <p>Minimum demand is 1272 MW (60%); peak demand is 2120 MW. Each row solves both periods independently with α₁ = 1. LF loss includes its two lines and both complete B2B converter pairs; total loss also includes HVAC and MTDC losses. Values below are local continuous solutions under the declared limits.</p>
    {plan_table}
    <p><b>Design selection:</b> at λ = 0.1 MW/kV, the peak continuous solution is {design['continuous_peak_voltage_kv']:.3f} kV and 6.407 Hz. We select the next rating in the declared set {{138, 230, 345, 500}} kV: <b>345 kV</b>. This is a transparent selection rule, not a joint investment-cost optimum. All 72 operating points are re-solved at 345 kV and checked.</p>
    {plot_tag('planning_weights','Minimum and peak planning values nearly overlap because LF generation and DC transfer obligations stay fixed while HVAC demand varies.')}
    <p>Why voltage and frequency both matter: raising voltage lowers current for the same transfer, reducing I²R loss, but line charging rises with frequency and voltage. The optimizer balances conductor losses, reactive flows, converter losses and capability limits. Lower frequency alone does not guarantee the lowest total loss.</p>

    <h2 id="operation">3. Operation — three controlled comparisons</h2>
    <p><b>A:</b> fixed 10 Hz, feasible schedules prepared by a distance-based repair of a declared heuristic. The repair minimizes dispatch deviation, not loss; its effort is recorded separately. <b>B:</b> same obligations and 10 Hz, optimize active/reactive dispatch and generator voltage targets. <b>C:</b> the same decisions plus LF frequency. Every case uses the selected 345 kV LF design.</p>
    {peak_table}
    <div class="callout"><b>Interpretation:</b> dispatch gives most of the improvement. Over this synthetic day, B reduces loss energy by {summary['B_energy_reduction_pct']:.2f}% relative to A; C reduces it by {summary['C_energy_reduction_pct']:.2f}%. Allowing frequency to vary adds only <b>{summary['C_incremental_vs_B_pct']:.3f}%</b> relative to B ({summary['daily_loss_energy_mwh']['B']-summary['daily_loss_energy_mwh']['C']:.3f} MWh/day). This is a small incremental benefit for this particular case.</div>
    {plot_tag('hourly_losses','A/B/C total loss and loss as a fraction of demand. Hours are independent steady-state operating points; one-hour piecewise-constant values give the daily MWh totals.')}
    {plot_tag('demand_frequency',f'Case C frequency stays within {summary["C_frequency_range_hz"][0]:.4f}–{summary["C_frequency_range_hz"][1]:.4f} Hz. Fixed 200 MW LF generation makes near-constant frequency unsurprising. The hourly load profile is synthetic.')}
    {plot_tag('loss_breakdown','Peak hour 17: converters dominate total loss. These values include switching and transformer/reactor Joule losses without double counting.')}
    {plot_tag('voltage_envelope','Voltage bounds are met, but loss-only optimization moves some bus voltages to the upper bound. Lower loss does not automatically mean smaller voltage deviation.')}

    <h3>Explicit multi-objective extension</h3>
    <p>At peak demand we also solve <b>L + βΣ(|V| − 1)²</b> over the 22 AC buses. A larger β buys tighter voltage magnitudes at some loss cost. This is our extension; P1’s operational capacitor-switching penalty is absent because this reconstruction has no controllable capacitor banks.</p>
    {tradeoff_table}
    {plot_tag('multiobjective_tradeoffs','Sampled weighted local tradeoffs, not a complete or globally certified Pareto frontier. Planning uses voltage rating; operation uses bus-voltage deviation.')}
    <p class="small">Resistance-model check: keeping line resistance constant at its 10 Hz value (closer to P1’s approximation) gives {sensitivity['constant_R10']['frequency_hz']:.3f} Hz and {sensitivity['constant_R10']['loss_mw']:.4f} MW at peak, versus {sensitivity['frequency_dependent']['frequency_hz']:.3f} Hz and {sensitivity['frequency_dependent']['loss_mw']:.4f} MW with the disclosed R(F) interpolation. Both pass the independent circuit check.</p>

    <h2 id="dispatch">4. Dispatch — Table II analogue</h2>
    <p>Peak hour 17, Case C. Signs are <b>positive into the AC network</b>, so LF exports have negative AC-side P while HVAC receipts have positive P. The difference in power across each B2B pair is its converter loss. PCC voltage is in per unit on the applicable nominal base; the LF nominal rating is 345 kV.</p>
    {pair_table}
    <details><summary>All seven converter stations — A/B/C peak P, Q, voltage and loss</summary>{station_table}</details>
    <h3>Generator dispatch at peak demand</h3>
    {generator_table}
    <p class="small">HVAC buses 1 and 6 retain their reference angles; their P and Q are dependent balancing outputs. All generator outputs, including slack generators and fixed LF generation, are checked against declared P/Q/S limits. Controlled generator voltage targets are an explicit extension of P1’s fixed Vsch.</p>

    <h2 id="runtime">5. Computation — complete OPF and fixed PF work</h2>
    <p>Four backends execute the same numerical implementation. We time two independent peak OPFs (B and C) from fixed common starts, then a separate batch of exactly 72 saved PF points. Numba compiles <b>GS/Anderson plus converter, DC and outer PF coordination</b>; SLSQP remains Python/SciPy. Threads run independent scenarios, not bus updates inside a GS sweep.</p>
    {runtime_table}
    {plot_tag('runtime_comparison','Bars show medians; whiskers show observed min–max. Two workers for process/thread methods, native BLAS threads limited to one. Warm timings exclude pool startup and JIT/cache loading; OPF n=3, fixed-work PF n=7.')}
    <p>The {warm_seconds['python_process']:.3f} → {warm_seconds['numba_thread']:.3f} second two-OPF comparison is <b>our Python-process implementation versus our Numba-thread implementation</b>. It does not measure the authors’ runtime. This compile scope is wider than the earlier reconstruction benchmark, which compiled only the AC GS/Anderson kernel. The small 25-bus test does not establish scaling on larger hybrid networks.</p>
    <p class="small">Pools are persistent, case data preloaded, method order randomized and all process workers synchronized before warm batches. First runner invocations are saved separately; those timings include an actual first workload and are not isolated startup or universal cold-JIT measurements. Numerical outputs are checked after timing.</p>
    {plot_tag('hourly_optimizer_time','One recorded warm B/C optimization per hour in the serial Numba study. C starts from B’s solution, so this plot is descriptive and is not a fair cold-start comparison. Baseline preparation and A’s PF runtime are not plotted as zero-time solves.')}

    <h2 id="equations">6. Formulation and numerical method</h2>
    <p>Electrical equations use a 100 MVA per-unit base unless physical units are shown. Rated line-to-line voltage, per-unit bus voltage and DC voltage are different quantities. Power balance signs are consistent throughout. F denotes frequency; f in the rectangular voltage equations denotes the imaginary component of voltage.</p>
    <details open><summary>Derived state equations, constraints and objectives (24 equations)</summary>{equations}</details>
    <details><summary>Algorithm and implementation details</summary>
    <ol><li>Build the 19-bus HVAC admittance matrix with its AC tie; choose the two fixed HVAC reference phasors. For each candidate LF rating/frequency, rebuild its line and station parameters and 3-bus admittance.</li>
    <li>Apply controlled Pg, converter P/Q and voltage targets. Solve HVAC and LFAC PF by in-place Gauss–Seidel with Anderson history depth 8; keep PV magnitudes and reference buses fixed.</li>
    <li>Recover voltage-controller reactive injections and LF slack power. Calculate station terminal currents, voltages and losses from KCL/KVL.</li>
    <li>Solve the 3-terminal DC network with DC24 voltage fixed; update VSC2 P to balance it. Enforce the two B2B DC-link balances through dependent VSC4 and VSC7 P. Repeat AC/DC coordination until the interface residual is below 1e−9 p.u.; inner AC mismatch is below 1e−10 p.u.</li>
    <li>Calculate objective and all terminal/current/rating constraints. SLSQP updates unit-box-scaled controls using shared central finite differences (step 1e−4 in scaled coordinates). Acceptance requires solver success, physical feasibility and conservation; no global optimality claim.</li>
    <li>For reproducibility, no fastmath or bus-order change is used. Numba runs the same core function with <code>njit(cache=True, nogil=True)</code>. Only independent tasks are assigned to workers.</li></ol>
    </details>

    <h2 id="audit">7. Assumptions, provenance and verification</h2>
    {provenance}{ratings_table}{checks_table}
    <p>The adopted LF line model is nominal-pi with R(F) interpolated through 0, 10 and 60 Hz, and fixed physical L/C. The converter family uses physical R/L/C referred from the 500 kV / 10 Hz reference; its current and loss normalization remains tied to reference current. Changing design voltage presumes appropriate station voltage capability. These assumptions let us run a controlled steady-state experiment; they do not establish magnetic V/f limits, transformer/converter hardware costs, EMT stability, protection adequacy or OEM feasibility.</p>
    <p>There are no capacitor decisions, discrete design choices inside the optimizer, generator ramps, storage dynamics or ML inference. Planning applies operating line limits too, unlike P1’s planning formulation. We cannot call the solver a replication of P1’s mixed-integer predictor-corrector primal-dual interior-point method. Numerical feasibility is accepted to 1e−6 in constraint units; some active limits have tiny negative roundoff margins.</p>
    <h3>A defensible statement for the professor</h3>
    <div class="scope"><p>“I implemented a reduced-space, P1-style planning and operation study on a disclosed reconstruction of the P3 25-bus network, using sequential GS–Anderson PF. Ten planning and 72 operating points satisfy our declared steady-state limits. Dispatch provides most of the loss reduction; variable frequency gives a smaller incremental gain. Compiling the full PF core and threading independent OPF scenarios speeds up our implementation. The next validation step is the authors’ case data and actual equipment constraints.”</p></div>
    <h3>Source papers</h3>{reference_links}

    <h2 id="files">8. Reproduce and inspect the results</h2>
    <p>All figures are saved as SVG and PNG; tables and individual results are CSV/JSON. Code is in the parent directory. The <a href="../README.md">README</a> specifies dependencies, execution order and modeling differences.</p>
    <pre>python experiments/03_sequential_opf/run_study.py
python experiments/03_sequential_opf/run_tradeoff.py
python experiments/03_sequential_opf/benchmark_opf.py --opf-repeats 3 --pf-repeats 7
python experiments/03_sequential_opf/make_report.py</pre>
    {links}
    <footer>Generated from saved numerical results; no paper outputs were used as fitting targets. Source hashes, solver statuses and raw runtime samples are included. Scientific conclusions apply to the declared reconstructed 25-bus scenario.</footer>
    </main></body></html>"""
    (OUT / "report.html").write_text(doc, encoding="utf-8")
    environment = dict(
        python=sys.version,
        numpy=np.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
        llvmlite=llvmlite.__version__,
        matplotlib=matplotlib.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
        native_library_threads=1,
    )
    (OUT / "runtime_environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    paths = [
        HERE / "opf_core.py",
        HERE / "optimization.py",
        HERE / "run_study.py",
        HERE / "run_tradeoff.py",
        HERE / "benchmark_opf.py",
        HERE.parent / "01_numba_benchmark/benchmark.py",
        HERE.parent / "02_sequential_pf_25bus/model.py",
        HERE.parent / "02_sequential_pf_25bus/grid_utils.py",
        HERE.parent / "02_sequential_pf_25bus/data/reconstructed_25bus.json",
    ]
    (OUT / "source_hashes.json").write_text(
        json.dumps(
            {
                str(p.relative_to(HERE.parent)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in paths
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Created", OUT / "report.html", flush=True)


if __name__ == "__main__":
    main()
