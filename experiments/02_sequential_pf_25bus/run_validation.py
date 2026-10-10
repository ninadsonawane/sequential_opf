"""Reconstruct, solve, independently verify, and compare with paper tables."""

import p3_bootstrap as bootstrap  # noqa: F401
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import json
import platform
import time
import numpy as np
import scipy
import numba
from pypower.api import runpf, ppoption
from pypower.ext2int import ext2int
from pypower.makeYbus import makeYbus as pp_ybus
from pypower.makeSbus import makeSbus

from model import (
    BASE,
    ATTACH,
    build_case,
    sequential,
    simultaneous_reference,
    validate,
    converter,
    solve_impl,
    solve_native,
)
import published as p

OUT = bootstrap.HERE / "results"


def save_json(name, data):
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)

    (OUT / name).write_text(
        json.dumps(data, indent=2, default=encode), encoding="utf-8"
    )


def save_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def original_cases():
    """Unmodified inputs; recompute independent NR instead of saved solutions."""
    from source_cases import exact_source_args

    args, references = exact_source_args()
    rows = []
    for name, inputs, nr in zip(("Stagg5", "IEEE14"), args, references):
        py = solve_impl(*inputs)
        nb = solve_native(*inputs)
        dnr = float(max(abs(nb[0] - nr)))
        dp = float(max(abs(nb[0] - py[0])))
        assert max(py[2], nb[2]) < 1e-9 and max(dnr, dp) < 1e-7
        rows.append(
            dict(
                case=name,
                source_data_exact=True,
                p3_modified_case=False,
                python_iterations=int(py[1]),
                numba_iterations=int(nb[1]),
                max_mismatch_pu=float(max(py[2], nb[2])),
                independent_nr_phasor_error_pu=dnr,
                python_numba_phasor_difference_pu=dp,
            )
        )
    return rows


def component_checks(case):
    """Reported terminal operating points used ONLY as component test inputs.

    Not a validation of the reconstructed network; terminal results cannot
    simultaneously be used as fitted inputs and independent full-system targets.
    """
    rows = []
    literal = []
    for k in range(7):
        ss = complex(p.CONVERTERS["ps_mw"][k], p.CONVERTERS["qs_mvar"][k]) / BASE
        voltage_variants = [
            ("Table6/7", p.CONVERTERS["vm_pu"][k], p.CONVERTERS["angle_deg"][k]),
            ("Table4", p.VM[ATTACH[k]], p.ANGLE[ATTACH[k]]),
        ]
        for label, vm, ang in voltage_variants:
            v = vm * np.exp(1j * np.deg2rad(ang))
            c = converter(v, ss, case.par)
            metrics = dict(
                pc_mw=c["sc"].real * BASE,
                qc_mvar=c["sc"].imag * BASE,
                vc_pu=abs(c["vc"]),
                converter_angle_deg=np.rad2deg(np.angle(c["vc"])),
                switching_loss_mw=c["psw"] * BASE,
            )
            for metric, value in metrics.items():
                target = p.CONVERTERS[metric][k]
                rows.append(
                    dict(
                        station=k + 1,
                        terminal_input_source=label,
                        metric=metric,
                        reported=float(target),
                        computed=float(value),
                        error=float(value - target),
                        test_scope="converter only; reported terminal V,P,Q given",
                    )
                )
        if k == 0:
            vm = p.VM[4]
            ang = p.ANGLE[4]
            v = vm * np.exp(1j * np.deg2rad(ang))
            is_ = np.conj(ss / v)
            vf = v + case.par.zt * is_
            for label, yfilter in [
                ("Printed Eq9", 1 / case.par.zt),
                ("Literal Table3 Zf", 1 / (1j * 0.0887)),
                ("Capacitive KCL interpretation", 1j * 0.0887),
            ]:
                ic = is_ + yfilter * vf
                vc = vf + case.par.zc * ic
                sc = vc * np.conj(ic)
                a2 = case.par.a2_inv if sc.real >= 0 else case.par.a2_rect
                loss = case.par.a0 + case.par.a1 * abs(ic) + a2 * abs(ic) ** 2
                literal.append(
                    dict(
                        interpretation=label,
                        pc_mw=float(sc.real * BASE),
                        qc_mvar=float(sc.imag * BASE),
                        vc_pu=float(abs(vc)),
                        converter_current_pu=float(abs(ic)),
                        switching_loss_mw=float(loss * BASE),
                        published_vc_pu=1.058,
                        published_loss_mw=3.56,
                    )
                )
    return rows, literal


def main():
    OUT.mkdir(exist_ok=True)
    case = build_case()
    t = time.perf_counter()
    py = sequential(case)
    pytime = time.perf_counter() - t
    t = time.perf_counter()
    nb = sequential(case, engine="numba")
    first_native = time.perf_counter() - t
    reference = simultaneous_reference(case)
    audit = validate(case, py, reference)
    native_audit = validate(case, nb, reference)
    assert (
        audit["checks"]["numerically_verified"]
        and native_audit["checks"]["numerically_verified"]
    )
    d = max(
        np.max(abs(py["vh"] - nb["vh"])),
        np.max(abs(py["vl"] - nb["vl"])),
        max(abs(py["vd"] - nb["vd"])),
    )
    assert d < 1e-7
    save_json(
        "verification.json",
        dict(
            python=audit["checks"],
            numba=native_audit["checks"],
            python_numba_max_voltage_difference_pu=float(d),
            python_outer_iterations=py["outer_iterations"],
            numba_outer_iterations=nb["outer_iterations"],
            python_total_sweeps=py["total_sweeps"],
            numba_total_sweeps=nb["total_sweeps"],
            reference_nfev=reference["nfev"],
            paper_exact_reproduction=False,
        ),
    )
    save_json("power_balance.json", audit["power"])
    save_csv("converter_limits.csv", audit["limits"])
    save_csv("outer_iterations.csv", py["trace"])
    save_json("input_audit.json", p.AUDIT)
    save_json(
        "reconstructed_inputs.json",
        dict(
            settings=case.notes,
            converter=asdict(case.par),
            hvac={k: getattr(case.h, k) for k in ("bus", "gen", "branch")},
            lfac={k: getattr(case.l, k) for k in ("bus", "gen", "branch")},
            dc_ybus=case.dc_y,
        ),
    )
    save_json(
        "published_targets.json",
        dict(
            voltage_magnitude=p.VM,
            angle_deg=p.ANGLE,
            branch_flows=p.FLOWS,
            converters=p.CONVERTERS,
        ),
    )
    vols = np.r_[py["vh"], py["vl"]]
    vms = np.r_[abs(vols), py["vd"]]
    angles = np.rad2deg(np.angle(vols))
    bus = []
    for i in range(25):
        bus.append(
            dict(
                bus=i + 1,
                grid="HVAC" if i < 19 else "LFAC" if i < 22 else "DC",
                reported_vm_pu=float(p.VM[i]),
                computed_vm_pu=float(vms[i]),
                vm_error_pu=float(vms[i] - p.VM[i]),
                reported_angle_deg=float(p.ANGLE[i]) if i < 22 else None,
                computed_angle_deg=float(angles[i]) if i < 22 else None,
                angle_error_deg=float(angles[i] - p.ANGLE[i]) if i < 22 else None,
                vm_within_reported_rounding=bool(abs(vms[i] - p.VM[i]) <= 0.000500001),
                angle_within_reported_rounding=(
                    bool(abs(angles[i] - p.ANGLE[i]) <= 0.00500001) if i < 22 else None
                ),
            )
        )
    save_csv("paper_bus_comparison.csv", bus)
    allflows = {
        ("HVAC", f[0], f[1]): (f[2] * BASE, f[3] * BASE) for f in audit["flows_h"]
    }
    allflows.update(
        {
            ("LFAC", f[0] + 19, f[1] + 19): (f[2] * BASE, f[3] * BASE)
            for f in audit["flows_l"]
        }
    )
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        gd = -case.dc_y[i, j]
        vd = py["vd"]
        current = gd * (vd[i] - vd[j])
        allflows["DC", i + 23, j + 23] = (
            complex(vd[i] * current * BASE),
            complex(-vd[j] * current * BASE),
        )
    br = []
    for grid, f, t, pf, qf, pt, qt in p.FLOWS:
        sf, st = allflows[grid, f, t]
        for metric, computed, reported in [
            ("p_from_mw", sf.real, pf),
            ("p_to_mw", st.real, pt),
            ("q_from_mvar", sf.imag, qf),
            ("q_to_mvar", st.imag, qt),
        ]:
            if reported is None:
                continue
            br.append(
                dict(
                    grid=grid,
                    from_bus=f,
                    to_bus=t,
                    metric=metric,
                    reported=float(reported),
                    computed=float(computed),
                    error=float(computed - reported),
                )
            )
    save_csv("paper_branch_comparison.csv", br)
    cr = []
    for k, c in enumerate(py["converters"]):
        values = dict(
            ps_mw=py["ss"][k].real * BASE,
            qs_mvar=py["ss"][k].imag * BASE,
            pc_mw=c["sc"].real * BASE,
            qc_mvar=c["sc"].imag * BASE,
            vm_pu=abs(vols[ATTACH[k]]),
            angle_deg=np.rad2deg(np.angle(vols[ATTACH[k]])),
            vc_pu=abs(c["vc"]),
            converter_angle_deg=np.rad2deg(np.angle(c["vc"])),
            switching_loss_mw=c["psw"] * BASE,
        )
        for metric, value in values.items():
            cr.append(
                dict(
                    station=k + 1,
                    metric=metric,
                    reported=p.CONVERTERS[metric][k],
                    computed=float(value),
                    error=float(value - p.CONVERTERS[metric][k]),
                )
            )
    save_csv("paper_converter_comparison.csv", cr)
    components, literal = component_checks(case)
    save_csv("converter_component_checks.csv", components)
    save_csv("notation_interpretation_comparison.csv", literal)
    sourcechecks = original_cases()
    save_json("exact_public_case_checks.json", sourcechecks)
    np.savez(
        OUT / "solution.npz",
        vh=py["vh"],
        vl=py["vl"],
        vd=py["vd"],
        ss=py["ss"],
        vh_reference=reference["vh"],
        vl_reference=reference["vl"],
        vd_reference=reference["vd"],
    )
    # Untuned sensitivity makes the DC base ambiguity visible, not a fitted fix.
    alternate = build_case(pole_factor=2)
    ar = sequential(alternate)
    aref = simultaneous_reference(alternate)
    aa = validate(alternate, ar, aref)
    assert aa["checks"]["numerically_verified"]
    save_json(
        "dc_pole_assumption_sensitivity.json",
        dict(
            reason="P3 presents mono/bipole equations but does not identify case configuration.",
            single_effective_pole=dict(
                dc_voltage=py["vd"],
                vsc2_ps_mw=py["ss"][1].real * BASE,
                dc_line_loss_mw=audit["power"]["dc_line_loss_mw"],
            ),
            two_effective_poles=dict(
                dc_voltage=ar["vd"],
                vsc2_ps_mw=ar["ss"][1].real * BASE,
                dc_line_loss_mw=aa["power"]["dc_line_loss_mw"],
            ),
        ),
    )
    env = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
        platform=platform.platform(),
        first_native_call_seconds=first_native,
        untimed_python_validation_seconds=pytime,
        numba_flags=dict(cache=True, nogil=True, fastmath=False),
        source_sha256={
            n: hashlib.sha256((bootstrap.HERE / n).read_bytes()).hexdigest()
            for n in ["model.py", "published.py", "data/reconstructed_25bus.json"]
        },
    )
    save_json("environment.json", env)
    summary = dict(
        full_system_numerically_verified=True,
        exact_paper_reproduction=False,
        topology="5 HVAC + 14 HVAC + 3 LFAC + 3 DC = 25 buses, seven VSCs",
        outer_iterations=py["outer_iterations"],
        total_sweeps=py["total_sweeps"],
        lfac_line_loss_mw=audit["power"]["lfac_line_loss_mw"],
        paper_lfac_line_loss_mw=2.03,
        lfac_angle20_deg=float(angles[19]),
        paper_lfac_angle20_deg=2.35,
        max_hvac_angle_error_deg=float(max(abs(angles[:19] - p.ANGLE[:19]))),
        max_hvac_vm_error_pu=float(max(abs(vms[:19] - p.VM[:19]))),
        max_lfac_vm_error_pu=float(max(abs(vms[19:22] - p.VM[19:22]))),
        max_dc_vm_error_pu=float(max(abs(vms[22:] - p.VM[22:]))),
        converter_current_violations=[
            x["station"] for x in audit["limits"] if not x["current_limit_ok"]
        ],
        voltage_magnitudes_within_table_rounding=sum(
            x["vm_within_reported_rounding"] for x in bus
        ),
        exact_public_cases_checked=sourcechecks,
    )
    save_json("summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
