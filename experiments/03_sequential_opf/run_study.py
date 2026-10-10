"""Actual planning table, 24-hour A/B/C study, dispatch tables and checks."""

import opf_bootstrap
import csv
import json
import hashlib
import time
from pathlib import Path
import numpy as np
from optimization import (
    Evaluator,
    baseline,
    optimize,
    restore_feasibility,
    reference_circuit,
    PROFILE,
    GEN_BUS,
    GEN_PMAX,
    GEN_QMAX,
    GEN_SMAX,
    LOW,
    HIGH,
    CONVERTER_CURRENT_REF_LIMIT,
)
from model import build_case
from opf_core import ATTACH

HERE = opf_bootstrap.HERE
OUT = HERE / "results"


def save_json(name, obj):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, complex):
            return {"real": x.real, "imag": x.imag}
        raise TypeError(type(x).__name__)

    (OUT / name).write_text(
        json.dumps(obj, indent=2, default=default), encoding="utf-8"
    )


def save_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def demand_accept(record, label):
    if not record["accepted"]:
        r = record["result"]
        bad = []
        if "constraint_labels" in r:
            bad = [
                (k, float(x))
                for k, x in zip(r["constraint_labels"], r["constraints"])
                if x < 0
            ]
        raise RuntimeError(
            f"{label}: {record['message']}; constraints {bad}; x {r.get('x')}"
        )


def compact(record):
    return {k: v for k, v in record.items() if k != "result"}


def main():
    OUT.mkdir(exist_ok=True)
    base = build_case()
    # Regression to the previous reconstructed case at f=10Hz/V=500kV.
    legacy = np.r_[
        np.array([0.4, 0.4, 0, 0, 0, 0.4, -0.3, 0.05, -0.3, -0.1, 0.1, 1, 1]),
        base.h.bus[GEN_BUS, 7],
    ]
    old = np.load(HERE.parent / "02_sequential_pf_25bus/recorded_results/solution.npz")
    ev = Evaluator(base_case=base)
    clock = time.perf_counter()
    r = ev.evaluate(legacy)
    first_native = time.perf_counter() - clock
    legacy_errors = {
        k: float(np.max(abs(r[k] - old[k]))) for k in ("vh", "vl", "vd", "ss")
    }
    assert max(legacy_errors.values()) < 1e-7
    py = Evaluator(engine="python", base_case=base).evaluate(legacy)
    assert max(np.max(abs(r[k] - py[k])) for k in ("vh", "vl", "vd", "ss")) < 1e-7
    save_json(
        "legacy_regression.json",
        dict(
            errors=legacy_errors,
            first_native_call_seconds=first_native,
            common_base_case_match=True,
            notes="Equipment-rating limits differ; equations agree at the same inputs.",
        ),
    )
    weights = [0, 0.1, 0.2, 0.3, 0.4]
    planning = []
    planning_details = []
    for label, profile in [
        ("minimum", float(PROFILE.min())),
        ("peak", float(PROFILE.max())),
    ]:
        e = Evaluator(profile, base_case=base)
        seed = restore_feasibility(e, baseline(profile))
        demand_accept(seed, "planning feasible initialization")
        start = seed["result"]["x"]
        for weight in weights:
            o = optimize(e, start, planning=True, voltage_weight=weight)
            demand_accept(o, f"planning {label} lambda {weight}")
            rr = o["result"]
            start = rr["x"]
            check = reference_circuit(e, start)
            assert check["verified"]
            planning.append(
                dict(
                    load_period=label,
                    load_fraction=profile,
                    alpha1=1.0,
                    alpha2_mw_per_kv=weight,
                    voltage_rating_kv=float(start[12] * 500),
                    frequency_hz=float(start[11] * 10),
                    lf_grid_loss_mw=rr["lfac_total_loss_mw"],
                    total_loss_mw=rr["loss_mw"],
                    total_loss_pct=rr["loss_mw"] / rr["demand_mw"] * 100,
                    objective=o["objective"],
                    feasible=rr["feasible"],
                    min_constraint_margin=rr["min_constraint"],
                    pf_evaluations=o["pf_evaluations"],
                    iterations=o["iterations"],
                    wall_seconds=o["wall_seconds"],
                    independent_max_voltage_error_pu=check["max_voltage_error_pu"],
                )
            )
            planning_details.append(
                dict(
                    period=label,
                    weight=weight,
                    x=start,
                    optimizer=compact(o),
                    verification=check,
                )
            )
            print(
                "planning",
                label,
                weight,
                "V",
                round(start[12] * 500, 3),
                "f",
                round(start[11] * 10, 3),
                "loss",
                round(rr["loss_mw"], 5),
                "verified",
                flush=True,
            )
    save_csv("table1_planning.csv", planning)
    save_json("planning_details.json", planning_details)
    peak_chosen = next(
        r
        for r in planning
        if r["load_period"] == "peak" and r["alpha2_mw_per_kv"] == 0.1
    )
    # Choose a standard design rating at/above the continuous planning result.
    design_candidates = [138.0, 230.0, 345.0, 500.0]
    selected = next(
        k for k in design_candidates if k >= peak_chosen["voltage_rating_kv"] - 1e-6
    )
    # Check the selected design explicitly through all scenarios; rounding alone
    # does not certify feasibility.
    save_json(
        "design_selection.json",
        dict(
            planning_weight=0.1,
            continuous_peak_voltage_kv=peak_chosen["voltage_rating_kv"],
            candidate_voltage_ratings_kv=design_candidates,
            selected_rating_kv=selected,
            fixed_frequency_baseline_hz=10,
            rule="Next declared design voltage not below peak planning result; all hourly solutions rechecked.",
        ),
    )
    rows = []
    dispatch = []
    generators = []
    buses = []
    constraints = []
    starts = []
    states = []
    preparation = []
    optimizer_records = []
    verifications = []
    multistart = []
    for hour, profile in enumerate(PROFILE):
        e = Evaluator(float(profile), base_case=base)
        target = baseline(float(profile), kv=selected)
        a = restore_feasibility(e, target)
        demand_accept(a, f"A baseline hour {hour}")
        b = optimize(e, a["result"]["x"], free_frequency=False)
        demand_accept(b, f"B hour {hour}")
        c = optimize(e, b["result"]["x"], free_frequency=True)
        demand_accept(c, f"C hour {hour}")
        # Additional starts at the two planning extremes. Local optimality only.
        if hour in (int(np.argmin(PROFILE)), int(np.argmax(PROFILE))):
            for freq in (5.0, 20.0):
                x = c["result"]["x"].copy()
                x[11] = freq / 10
                other = optimize(e, x, free_frequency=True)
                multistart.append(
                    dict(
                        hour=hour,
                        start_frequency_hz=freq,
                        accepted=other["accepted"],
                        loss_mw=other["result"]["loss_mw"],
                        message=other["message"],
                        difference_from_primary_mw=other["result"]["loss_mw"]
                        - c["result"]["loss_mw"],
                    )
                )
                if (
                    other["accepted"]
                    and other["result"]["loss_mw"] < c["result"]["loss_mw"] - 1e-6
                ):
                    c = other
        assert b["result"]["loss_mw"] <= a["result"]["loss_mw"] + 1e-5
        assert c["result"]["loss_mw"] <= b["result"]["loss_mw"] + 1e-5
        starts.append(
            dict(
                hour=hour,
                profile=float(profile),
                A=a["result"]["x"].tolist(),
                B_start=a["result"]["x"].tolist(),
                C_start=b["result"]["x"].tolist(),
            )
        )
        preparation.append(
            dict(
                hour=hour,
                profile=profile,
                loss_before_repair_mw=e.evaluate(target)["loss_mw"],
                initial_heuristic_feasible=e.evaluate(target)["feasible"],
                feasible_repair_objective=a["objective"],
                repair_wall_seconds=a["wall_seconds"],
                repair_pf_evaluations=a["pf_evaluations"],
            )
        )
        for name, record in [("A", a), ("B", b), ("C", c)]:
            rr = record["result"]
            x = rr["x"]
            assert rr["feasible"] and abs(rr["power_balance_error_mw"]) < 1e-5
            assert max(rr["ac_residual_pu"], rr["dc_interface_residual_pu"]) < 1e-8
            rows.append(
                dict(
                    hour=hour,
                    load_fraction=float(profile),
                    case=name,
                    demand_mw=rr["demand_mw"],
                    voltage_rating_kv=selected,
                    frequency_hz=float(x[11] * 10),
                    total_loss_mw=rr["loss_mw"],
                    loss_pct=rr["loss_mw"] / rr["demand_mw"] * 100,
                    hvac_line_loss_mw=rr["hvac_loss_mw"],
                    lfac_line_loss_mw=rr["lfac_line_loss_mw"],
                    dc_line_loss_mw=rr["dc_line_loss_mw"],
                    converter_loss_mw=float(sum(rr["converter_loss_mw"])),
                    voltage_deviation=rr["voltage_deviation"],
                    min_voltage_pu=rr["min_voltage_pu"],
                    max_voltage_pu=rr["max_voltage_pu"],
                    feasible=rr["feasible"],
                    min_constraint_margin=rr["min_constraint"],
                    ac_residual_pu=rr["ac_residual_pu"],
                    dc_interface_residual_pu=rr["dc_interface_residual_pu"],
                    power_balance_error_mw=rr["power_balance_error_mw"],
                    solve_wall_seconds=record["wall_seconds"] if name != "A" else 0.0,
                    pf_evaluations=record["pf_evaluations"] if name != "A" else 0,
                )
            )
            states.append(dict(hour=hour, case=name, x=x))
            vv = np.r_[rr["vh"], rr["vl"]]
            for k in range(7):
                dispatch.append(
                    dict(
                        hour=hour,
                        case=name,
                        station=f"VSC{k+1}",
                        pcc_bus=int(ATTACH[k] + 1),
                        p_ac_mw=float(rr["ss"][k].real * 100),
                        q_ac_mvar=float(rr["ss"][k].imag * 100),
                        terminal_voltage_pu=float(abs(vv[ATTACH[k]])),
                        terminal_angle_deg=float(np.rad2deg(np.angle(vv[ATTACH[k]]))),
                        pc_mw=float(rr["sc"][k].real * 100),
                        qc_mvar=float(rr["sc"][k].imag * 100),
                        converter_voltage_pu=float(abs(rr["vc"][k])),
                        converter_current_reference_pu=float(rr["ic_reference_pu"][k]),
                        station_loss_mw=float(rr["converter_loss_mw"][k]),
                    )
                )
            for k, bus in enumerate(np.r_[GEN_BUS + 1, 20]):
                generators.append(
                    dict(
                        hour=hour,
                        case=name,
                        bus=int(bus),
                        p_generation_mw=float(rr["generation_mw"][k]),
                        q_generation_mvar=float(rr["generation_mvar"][k]),
                    )
                )
            for k, vv0 in enumerate(np.r_[vv, rr["vd"].astype(complex)]):
                buses.append(
                    dict(
                        hour=hour,
                        case=name,
                        bus=k + 1,
                        voltage_pu=float(abs(vv0)),
                        angle_deg=float(np.rad2deg(np.angle(vv0))) if k < 22 else None,
                    )
                )
            for label, margin in zip(rr["constraint_labels"], rr["constraints"]):
                constraints.append(
                    dict(
                        hour=hour,
                        case=name,
                        constraint=label,
                        normalized_margin=float(margin),
                        satisfied=bool(margin >= -1e-6),
                    )
                )
            optimizer_records.append(
                dict(hour=hour, case=name, optimizer=compact(record))
            )
            if hour in (int(np.argmin(PROFILE)), int(np.argmax(PROFILE))):
                check = reference_circuit(e, x)
                assert check["verified"]
                verifications.append(dict(hour=hour, case=name, **check))
        print(
            "hour",
            hour,
            "loss A/B/C",
            *[round(z["result"]["loss_mw"], 5) for z in (a, b, c)],
            "fC",
            round(c["result"]["x"][11] * 10, 4),
            "feasible",
            flush=True,
        )
    save_csv("hourly_results.csv", rows)
    save_csv("baseline_feasibility_preparation.csv", preparation)
    save_csv("converter_dispatch_all_hours.csv", dispatch)
    save_csv("generator_dispatch_all_hours.csv", generators)
    peak_hour = int(np.argmax(PROFILE))
    save_csv(
        "table2_converter_dispatch_peak.csv",
        [r for r in dispatch if r["hour"] == peak_hour],
    )
    save_csv(
        "table2_generator_dispatch_peak.csv",
        [r for r in generators if r["hour"] == peak_hour],
    )
    save_csv("bus_voltages_all_hours.csv", buses)
    save_csv("constraint_audit_all_hours.csv", constraints)
    save_json("optimization_starts.json", starts)
    save_json("solution_controls.json", states)
    save_json("optimizer_records.json", optimizer_records)
    save_json("independent_validation.json", verifications)
    save_json("multistart_checks.json", multistart)
    # Additional operational objective tradeoff, separate from P1-like B/C.
    pe = Evaluator(1.0, base_case=base)
    seed = next(s["x"] for s in states if s["hour"] == peak_hour and s["case"] == "C")
    pareto = []
    pareto_controls = []
    for beta in (0.0, 1000.0, 5000.0):
        po = optimize(pe, seed, free_frequency=True, voltage_deviation_weight=beta)
        demand_accept(po, f"operational voltage tradeoff beta {beta}")
        pr = po["result"]
        seed = pr["x"]
        pv = reference_circuit(pe, seed)
        assert pv["verified"]
        pareto.append(
            dict(
                beta_mw_per_pu_squared=beta,
                total_loss_mw=pr["loss_mw"],
                voltage_deviation_pu_squared=pr["voltage_deviation"],
                frequency_hz=float(seed[11] * 10),
                min_voltage_pu=pr["min_voltage_pu"],
                max_voltage_pu=pr["max_voltage_pu"],
                objective=po["objective"],
                feasible=pr["feasible"],
                wall_seconds=po["wall_seconds"],
                independent_max_voltage_error_pu=pv["max_voltage_error_pu"],
            )
        )
        pareto_controls.append(dict(beta=beta, x=seed, optimizer=compact(po)))
    save_csv("operational_multiobjective_tradeoff.csv", pareto)
    save_json("operational_multiobjective_controls.json", pareto_controls)
    totals = {
        name: sum(r["total_loss_mw"] for r in rows if r["case"] == name)
        for name in ("A", "B", "C")
    }
    peak = {
        name: next(r for r in rows if r["case"] == name and r["hour"] == peak_hour)
        for name in ("A", "B", "C")
    }
    summary = dict(
        all_72_operating_points_feasible=True,
        planning_points=10,
        selected_rating_kv=selected,
        original_P1_reproduction=False,
        original_P3_exact_case=False,
        balanced_positive_sequence=True,
        daily_loss_energy_mwh=totals,
        peak_hour=peak_hour,
        peak=peak,
        B_energy_reduction_pct=(1 - totals["B"] / totals["A"]) * 100,
        C_energy_reduction_pct=(1 - totals["C"] / totals["A"]) * 100,
        C_incremental_vs_B_pct=(1 - totals["C"] / totals["B"]) * 100,
        C_frequency_range_hz=[
            min(r["frequency_hz"] for r in rows if r["case"] == "C"),
            max(r["frequency_hz"] for r in rows if r["case"] == "C"),
        ],
        max_conservation_error_mw=max(abs(r["power_balance_error_mw"]) for r in rows),
        max_independent_voltage_error_pu=max(
            r["max_voltage_error_pu"] for r in verifications
        ),
        optimizer="SciPy SLSQP; finite differences; reduced space; local optima only",
    )
    save_json("summary.json", summary)
    save_json(
        "assumptions.json",
        dict(
            peak_demand_mw=2120,
            load_profile=PROFILE,
            generator_buses=np.r_[GEN_BUS + 1, 20],
            generator_pmax_mw=GEN_PMAX * 100,
            generator_qmax_mvar=GEN_QMAX * 100,
            generator_smax_mva=GEN_SMAX * 100,
            converter_ratings_mva=[200] * 7,
            converter_current_ref_limit_pu=CONVERTER_CURRENT_REF_LIMIT,
            bus_voltage_limits_pu=[0.94, 1.06],
            dc_voltage_limits_pu=[0.95, 1.05],
            converter_voltage_limits_pu=[0.85, 1.2],
            hvac_line_amp_limit=1500,
            lfac_line_amp_limit=1000,
            dc_line_amp_limit=1000,
            hvac_line_power_limit_mw=1000,
            lfac_line_power_limit_mw=600,
            lf_generator_fixed_mw=200,
            dc_transfer_schedules_mw={"VSC1": -200, "VSC3": 100},
            resistance_model="Piecewise linear through Rdc=.01325,R10=.01345,R60=.0154 ohm/km; extrapolation below10Hz disclosed.",
            planning_voltage_range_kv=[69, 500],
            frequency_range_hz=[1, 60],
            shunt_capacitor_controls=False,
            capacitor_switching_objective=False,
            intertemporal_ramps=False,
            dynamic_EMT_or_V_over_f_magnetic_checks=False,
            demand_profile_origin="Declared synthetic coincident profile, not utility measurements or P1 recovered data.",
            equipment_rating_origin="Declared synthetic benchmark ratings. No claimed author/OEM/utility ratings.",
            baseline_A="Feasibility-only distance-based dispatch repair toward a declared heuristic; squared dispatch deviation objective, not loss minimization or a certified closest projection.",
            planning_scope="Same operating limits enforced in planning and operation; stricter than P1 planning which omits line limits.",
            lf_converter_model="Fixed physical R,L,C referred from500kV/10Hz; candidate device voltage rating follows design voltage; loss/current normalized to500kV reference current.",
            variable_control_names=[
                "Pg_bus2",
                "Pg_bus7",
                "Pg_bus8",
                "Pg_bus11",
                "Pg_bus13",
                "P_VSC5",
                "Q_VSC1",
                "Q_VSC3",
                "Q_VSC4",
                "Q_VSC5",
                "Q_VSC7",
                "f_LF_over10",
                "rated_V_LF_over500",
                "Vm_gen1",
                "Vm_gen2",
                "Vm_gen6",
                "Vm_gen7",
                "Vm_gen8",
                "Vm_gen11",
                "Vm_gen13",
            ],
            generator_voltage_controls="Extension of P1 fixed Vsch; bounded .96--1.04 pu to allow feasible reactive dispatch over the synthetic load profile.",
            control_lower=LOW,
            control_upper=HIGH,
        ),
    )
    save_json(
        "source_hashes.json",
        {
            str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                HERE / "opf_core.py",
                HERE / "optimization.py",
                HERE / "run_study.py",
            ]
        },
    )
    print("COMPLETE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
