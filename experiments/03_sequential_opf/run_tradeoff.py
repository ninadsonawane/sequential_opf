"""Additional, explicitly non-P1 voltage tradeoff and R(f) sensitivity."""

import opf_bootstrap
import json
import csv
from optimization import Evaluator, optimize, reference_circuit
from run_study import save_csv, save_json, demand_accept, compact

OUT = opf_bootstrap.HERE / "results"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    states = json.loads((OUT / "solution_controls.json").read_text())
    seed = next(
        s["x"] for s in states if s["hour"] == summary["peak_hour"] and s["case"] == "C"
    )
    e = Evaluator(1.0)
    rows = []
    controls = []
    for beta in (0.0, 1000.0, 5000.0):
        o = optimize(e, seed, free_frequency=True, voltage_deviation_weight=beta)
        demand_accept(o, f"voltage objective beta {beta}")
        r = o["result"]
        seed = r["x"]
        check = reference_circuit(e, seed)
        assert check["verified"]
        rows.append(
            dict(
                beta_mw_per_pu_squared=beta,
                total_loss_mw=r["loss_mw"],
                voltage_deviation_pu_squared=r["voltage_deviation"],
                frequency_hz=float(seed[11] * 10),
                min_voltage_pu=r["min_voltage_pu"],
                max_voltage_pu=r["max_voltage_pu"],
                objective=o["objective"],
                feasible=r["feasible"],
                wall_seconds=o["wall_seconds"],
                independent_max_voltage_error_pu=check["max_voltage_error_pu"],
            )
        )
        controls.append(
            dict(beta=beta, x=seed, optimizer=compact(o), verification=check)
        )
        print(
            "beta",
            beta,
            "loss",
            r["loss_mw"],
            "voltage deviation",
            r["voltage_deviation"],
            flush=True,
        )
    save_csv("operational_multiobjective_tradeoff.csv", rows)
    save_json("operational_multiobjective_controls.json", controls)
    original = next(
        s["x"] for s in states if s["hour"] == summary["peak_hour"] and s["case"] == "C"
    )
    e0 = Evaluator(1.0, resistance_model=0)
    constant = optimize(e0, original, free_frequency=True)
    demand_accept(constant, "constant resistance sensitivity")
    reference = reference_circuit(e0, constant["result"]["x"])
    assert reference["verified"]
    save_json(
        "resistance_model_sensitivity.json",
        dict(
            frequency_dependent=dict(
                frequency_hz=summary["peak"]["C"]["frequency_hz"],
                loss_mw=summary["peak"]["C"]["total_loss_mw"],
            ),
            constant_R10=dict(
                frequency_hz=float(constant["result"]["x"][11] * 10),
                loss_mw=constant["result"]["loss_mw"],
                accepted=constant["accepted"],
                verification=reference,
            ),
            note="Constant-R is closer to the P1 approximation; primary study uses interpolated P3-derived R(f). Neither is an exact replication of P1 data.",
        ),
    )


if __name__ == "__main__":
    main()
