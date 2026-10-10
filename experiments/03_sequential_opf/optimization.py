"""Feasibility-constrained, locally solved P1-style planning/operation."""

import opf_bootstrap
from collections import OrderedDict
from dataclasses import dataclass
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize, root
from model import build_case, branch_power, Grid, make_ybus
from opf_core import core_impl, core_native, parameters, lf_admittance, ATTACH, PG_BUS

BASE = 100.0
PROFILE = np.array(
    [
        0.72,
        0.68,
        0.64,
        0.62,
        0.60,
        0.62,
        0.70,
        0.78,
        0.84,
        0.88,
        0.90,
        0.91,
        0.92,
        0.90,
        0.88,
        0.92,
        0.98,
        1.0,
        0.99,
        0.95,
        0.90,
        0.85,
        0.80,
        0.76,
    ]
)
GEN_BUS = np.array([0, 1, 5, 6, 7, 10, 12], dtype=int)
# Ratings below are declared synthetic benchmark assumptions, not source ratings.
GEN_PMAX = np.array([1250, 1500, 1662, 700, 500, 500, 500, 250]) / BASE
GEN_QMAX = np.r_[0.75 * GEN_PMAX[:7], 4.0]
GEN_SMAX = np.sqrt(GEN_PMAX**2 + GEN_QMAX**2)
CONVERTER_CURRENT_REF_LIMIT = 2.1  # 1.05 * illustrative 200MVA / 100MVA.

LOW = np.r_[
    np.array([0, 0, 0, 0, 0, 0, -1.5, -1.5, -1.5, -1.5, -1.5, 0.1, 69 / 500]),
    np.ones(7) * 0.96,
]
HIGH = np.r_[
    np.array([15, 7, 5, 5, 5, 2, 1.5, 1.5, 1.5, 1.5, 1.5, 6.0, 1.0]), np.ones(7) * 1.04
]


def baseline(profile=1.0, kv=500.0, frequency=10.0):
    return np.array(
        [
            4 * profile,
            2 * profile,
            2 * profile,
            profile,
            profile,
            1.0,
            -0.3,
            0.05,
            -0.3,
            -0.1,
            0.1,
            frequency / 10,
            kv / 500,
            1.01,
            1,
            1,
            0.99,
            1.02,
            0.98,
            1.02,
        ]
    )


class Evaluator:
    def __init__(self, profile=1.0, engine="numba", resistance_model=1, base_case=None):
        self.profile = profile
        self.engine = engine
        self.resistance_model = resistance_model
        self.case = build_case() if base_case is None else base_case
        h = self.case.h
        y = csr_matrix(make_ybus(h))
        y.sort_indices()
        self.fixed = (
            y.indptr.astype(np.int64),
            y.indices.astype(np.int64),
            y.data,
            y.diagonal(),
            h.bus[:, 1].astype(np.int64),
            h.bus[:, 7].copy(),
            (h.bus[:, 2] + 1j * h.bus[:, 3]) / BASE,
            self.case.dc_y,
        )
        self.yh = y
        self.calls = 0
        self.pf_seconds = 0.0
        self.failures = 0
        self.cache = OrderedDict()
        self.h_from = h.branch[:, 0].astype(int) - 1
        self.h_to = h.branch[:, 1].astype(int) - 1
        self.h_z = h.branch[:, 2] + 1j * h.branch[:, 3]
        self.h_b = h.branch[:, 4]
        self.h_amp_base = np.full(len(h.branch), BASE * 1000 / (np.sqrt(3) * 345))
        self.h_amp_base[-1] = BASE * 1000 / (np.sqrt(3) * 500)
        self.constraint_count = None

    def evaluate(self, x):
        x = np.asarray(x, float)
        key = x.tobytes()
        if key in self.cache:
            return self.cache[key]
        fn = core_native if self.engine == "numba" else core_impl
        start = time.perf_counter()
        self.calls += 1
        try:
            out = fn(*self.fixed, x, self.profile, self.resistance_model, 1e-10, 1e-9)
            self.pf_seconds += time.perf_counter() - start
            if not out[-1] or not np.all(np.isfinite(out[0])):
                raise RuntimeError("PF failed")
            result = self.account(x, out)
        except (ValueError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError):
            self.failures += 1
            if self.constraint_count is None:
                raise
            result = dict(
                converged=False,
                loss_mw=1e5,
                constraints=np.full(self.constraint_count, -1000.0),
                voltage_deviation=1e5,
            )
        if self.constraint_count is None:
            self.constraint_count = len(result["constraints"])
        self.cache[key] = result
        if len(self.cache) > 1024:
            self.cache.popitem(last=False)
        return result

    def account(self, x, out):
        # Keep physical loss accounting outside the solver: agreement of the
        # nonlinear residual alone does not prove conservation or feasibility.
        vh, vl, vd, ss, ic, vc, sc, sw, joule, pdc, acerr, dcerr, it, sweeps, ok = out
        sh = vh * np.conj(self.yh @ vh)
        sf = (vh[self.h_from] - vh[self.h_to]) / self.h_z
        ihf = sf + 1j * self.h_b / 2 * vh[self.h_from]
        iht = -sf + 1j * self.h_b / 2 * vh[self.h_to]
        fh = vh[self.h_from] * np.conj(ihf)
        th = vh[self.h_to] * np.conj(iht)
        ld, _, zl, bl = lf_admittance(x[11], x[12], self.resistance_model)
        ilf = np.array(
            [
                (vl[0] - vl[1]) / zl + 1j * bl / 2 * vl[0],
                (vl[0] - vl[2]) / zl + 1j * bl / 2 * vl[0],
            ]
        )
        ilt = np.array(
            [
                (vl[1] - vl[0]) / zl + 1j * bl / 2 * vl[1],
                (vl[2] - vl[0]) / zl + 1j * bl / 2 * vl[2],
            ]
        )
        fl = vl[0] * np.conj(ilf)
        tl = vl[1:] * np.conj(ilt)
        yl = np.array([[ld[0], ld[1], ld[2]], [ld[3], ld[4], 0], [ld[5], 0, ld[6]]])
        sl = vl * np.conj(yl @ vl)
        conv = np.zeros(19, complex)
        for k in range(5):
            conv[ATTACH[k]] += ss[k]
        demand = self.fixed[6] * self.profile
        needed = sh + demand - conv
        generation = np.r_[needed[GEN_BUS], sl[0]]
        pg = generation.real
        qg = generation.imag
        lh = float(sum((fh + th).real) * BASE)
        ll = float(sum((fl + tl).real) * BASE)
        ld_loss = float(vd @ (self.case.dc_y @ vd) * BASE)
        converter_loss = float(sum(sw + joule) * BASE)
        loss = lh + ll + ld_loss + converter_loss
        amps_h = np.maximum(abs(ihf), abs(iht)) * self.h_amp_base
        amps_l = (
            np.maximum(abs(ilf), abs(ilt)) * BASE * 1000 / (np.sqrt(3) * 500 * x[12])
        )
        edge = np.array([[0, 1], [0, 2], [1, 2]])
        dc_i = np.array([-self.case.dc_y[i, j] * (vd[i] - vd[j]) for i, j in edge])
        dc_amps = abs(dc_i) * BASE * 1000 / 320
        ic_ref = abs(ic).copy()
        ic_ref[5:] /= x[12]
        vm = abs(np.r_[vh, vl])
        cv = abs(vc)
        # Every inequality is written as a nonnegative margin. Voltage/P/Q
        # margins use per-unit quantities; current/S/line-power margins are
        # normalized by their declared rating. These are synthetic constraints,
        # not ratings recovered from the source paper or an equipment vendor.
        groups = [
            ("bus_voltage_low", vm - 0.94),
            ("bus_voltage_high", 1.06 - vm),
            ("dc_voltage_low", vd - 0.95),
            ("dc_voltage_high", 1.05 - vd),
            ("generator_p_low", pg),
            ("generator_p_high", GEN_PMAX - pg),
            ("generator_q_low", qg + GEN_QMAX),
            ("generator_q_high", GEN_QMAX - qg),
            ("generator_apparent_rating", 1 - (pg**2 + qg**2) / GEN_SMAX**2),
            ("converter_current", 1 - ic_ref**2 / CONVERTER_CURRENT_REF_LIMIT**2),
            ("converter_voltage_low", cv - 0.85),
            ("converter_voltage_high", 1.2 - cv),
            ("hvac_line_current", 1 - amps_h / 1500),
            ("lfac_line_current", 1 - amps_l / 1000),
            ("dc_line_current", 1 - dc_amps / 1000),
            (
                "hvac_line_power",
                1 - np.maximum(abs(fh.real), abs(th.real)) * BASE / 1000,
            ),
            (
                "lfac_line_power",
                1 - np.maximum(abs(fl.real), abs(tl.real)) * BASE / 600,
            ),
            (
                "transfer_direction",
                np.array([ss[1].real, ss[3].real, -ss[5].real, -ss[6].real]),
            ),
        ]
        labels = []
        values = []
        for name, vec in groups:
            for i, v in enumerate(vec):
                labels.append(f"{name}_{i+1}")
                values.append(v)
        cons = np.array(values)
        # Also check the bound on input controls at final acceptance.
        bound_violation = max(0.0, float(np.max(LOW - x)), float(np.max(x - HIGH)))
        return dict(
            converged=bool(ok),
            x=x.copy(),
            vh=vh,
            vl=vl,
            vd=vd,
            ss=ss,
            ic=ic,
            vc=vc,
            sc=sc,
            converter_loss_mw=(sw + joule) * BASE,
            switching_loss_mw=sw * BASE,
            station_joule_mw=joule * BASE,
            loss_mw=loss,
            hvac_loss_mw=lh,
            lfac_line_loss_mw=ll,
            lfac_total_loss_mw=float(ll + sum((sw + joule)[3:]) * BASE),
            dc_line_loss_mw=ld_loss,
            generation_mw=pg * BASE,
            generation_mvar=qg * BASE,
            demand_mw=float(demand.real.sum() * BASE),
            voltage_deviation=float(sum((vm - 1) ** 2)),
            min_voltage_pu=float(vm.min()),
            max_voltage_pu=float(vm.max()),
            constraints=cons,
            constraint_labels=labels,
            min_constraint=float(cons.min()),
            bound_violation=bound_violation,
            feasible=bool(cons.min() >= -1e-6 and bound_violation < 1e-7),
            ac_residual_pu=float(acerr),
            dc_interface_residual_pu=float(dcerr),
            outer_iterations=int(it),
            gs_sweeps=int(sweeps),
            power_balance_error_mw=float(
                pg.sum() * BASE - demand.real.sum() * BASE - loss
            ),
            ic_reference_pu=ic_ref,
            line_hvac_amp=amps_h,
            line_lfac_amp=amps_l,
            line_dc_amp=dc_amps,
            hvac_flow_from=fh * BASE,
            hvac_flow_to=th * BASE,
            lfac_flow_from=fl * BASE,
            lfac_flow_to=tl * BASE,
        )


def optimize(
    ev,
    start,
    planning=False,
    free_frequency=True,
    voltage_weight=0.0,
    voltage_deviation_weight=0.0,
    maxiter=160,
):
    start = np.clip(np.array(start, float), LOW, HIGH)
    idx = (
        np.arange(20)
        if planning
        else (
            np.r_[np.arange(12), np.arange(13, 20)]
            if free_frequency
            else np.r_[np.arange(11), np.arange(13, 20)]
        )
    )
    fixed = start.copy()
    ev.evaluate(start)
    lo = LOW[idx]
    width = HIGH[idx] - lo

    def full(z):
        x = fixed.copy()
        x[idx] = lo + width * z
        return x

    def objective(z):
        x = full(z)
        r = ev.evaluate(x)
        return (
            r["loss_mw"]
            + voltage_weight * x[12] * 500
            + voltage_deviation_weight * r["voltage_deviation"]
        )

    def constraints(z):
        return ev.evaluate(full(z))["constraints"]

    derivative_cache = {}

    def derivatives(z):
        key = z.tobytes()
        if key in derivative_cache:
            return derivative_cache[key]
        gf = np.zeros(len(z))
        gc = np.empty((ev.constraint_count, len(z)))
        # Bounded central differences in scaled box coordinates; at a bound
        # this becomes one-sided. Objective/constraint Jacobians share each
        # expensive PF evaluation. The PF state is not an optimization variable.
        for j in range(len(z)):
            zp = z.copy()
            zm = z.copy()
            zp[j] = min(1.0, z[j] + 1e-4)
            zm[j] = max(0.0, z[j] - 1e-4)
            gf[j] = (objective(zp) - objective(zm)) / (zp[j] - zm[j])
            gc[:, j] = (constraints(zp) - constraints(zm)) / (zp[j] - zm[j])
        derivative_cache.clear()
        derivative_cache[key] = (gf, gc)
        return gf, gc

    calls = ev.calls
    pf = ev.pf_seconds
    clock = time.perf_counter()
    sol = minimize(
        objective,
        (start[idx] - lo) / width,
        jac=lambda z: derivatives(z)[0],
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(idx),
        constraints={
            "type": "ineq",
            "fun": constraints,
            "jac": lambda z: derivatives(z)[1],
        },
        options={"maxiter": maxiter, "ftol": 1e-7, "disp": False},
    )
    elapsed = time.perf_counter() - clock
    x = full(sol.x)
    r = ev.evaluate(x)
    accepted = bool(
        sol.success
        and r["converged"]
        and r["feasible"]
        and abs(r["power_balance_error_mw"]) < 1e-5
    )
    return dict(
        result=r,
        accepted=accepted,
        solver_success=bool(sol.success),
        message=str(sol.message),
        iterations=int(sol.nit),
        nfev=int(sol.nfev),
        pf_evaluations=ev.calls - calls,
        pf_seconds=ev.pf_seconds - pf,
        wall_seconds=elapsed,
        objective=float(sol.fun),
        voltage_weight=voltage_weight,
        voltage_deviation_weight=voltage_deviation_weight,
    )


def restore_feasibility(ev, target):
    """Distance-based feasible dispatch repair; no loss-minimizing objective.

    Baseline A uses these precomputed schedules. Preparation effort is reported
    separately rather than labeling its infeasible heuristic inputs feasible.
    """
    target = np.clip(np.array(target, float), LOW, HIGH)
    r0 = ev.evaluate(target)
    if r0["feasible"]:
        return dict(
            result=r0,
            accepted=True,
            solver_success=True,
            message="Already feasible",
            iterations=0,
            pf_evaluations=0,
            wall_seconds=0.0,
            objective=0.0,
        )
    idx = np.r_[np.arange(11), np.arange(13, 20)]
    scales = np.r_[np.array([15, 7, 5, 5, 5]), 2, np.ones(5) * 1.5, np.ones(7) * 0.04]

    def full(z):
        x = target.copy()
        x[idx] = z
        return x

    def obj(z):
        return float(sum(((z - target[idx]) / scales) ** 2))

    def jac(z):
        return 2 * (z - target[idx]) / scales**2

    def cons(z):
        return ev.evaluate(full(z))["constraints"]

    before = ev.calls
    t = time.perf_counter()
    sol = minimize(
        obj,
        target[idx],
        jac=jac,
        method="SLSQP",
        bounds=list(zip(LOW[idx], HIGH[idx])),
        constraints={"type": "ineq", "fun": cons},
        options={"maxiter": 80, "ftol": 1e-7, "eps": 1e-4},
    )
    result = ev.evaluate(full(sol.x))
    # Baseline construction requires a feasible dispatch, not a certified
    # minimum-distance solution; preserve the optimizer status separately.
    return dict(
        result=result,
        accepted=bool(result["feasible"]),
        solver_success=bool(sol.success),
        message=str(sol.message),
        iterations=int(sol.nit),
        pf_evaluations=ev.calls - before,
        wall_seconds=time.perf_counter() - t,
        objective=float(sol.fun),
    )


def reference_circuit(ev, x):
    """Independent expanded-circuit simultaneous solve, separate flat start."""
    case = ev.case
    zt, zc, bf = parameters(x[11], x[12])
    hdemand = ev.fixed[6] * ev.profile
    hbase = -hdemand.copy()
    for k, b in enumerate(PG_BUS):
        hbase[b] += x[k]
    ld, _, zl, bl = lf_admittance(x[11], x[12], ev.resistance_model)
    lb = case.l.bus.copy()
    lb[:, 9] = 500 * x[12]
    le = case.l.branch.copy()
    le[:, 2] = zl.real
    le[:, 3] = zl.imag
    le[:, 4] = bl
    lg = Grid("LFAC-opt", lb, case.l.gen.copy(), le)
    vm = np.r_[case.h.bus[:, 7], np.ones(3)]
    vm[GEN_BUS] = x[13:20]
    initial = np.r_[
        vm.astype(complex), vm[ATTACH].astype(complex), vm[ATTACH].astype(complex)
    ]
    n = len(initial)

    def unpack(z):
        zz = z[:n] + 1j * z[n : 2 * n]
        return zz[:22], zz[22:29], zz[29:], z[2 * n :]

    def equations(z):
        v, vf, vc, vd = unpack(z)
        ist = (vf - v[ATTACH]) / zt
        ic = (vc - vf) / zc
        ss = v[ATTACH] * np.conj(ist)
        sc = vc * np.conj(ic)
        ir = abs(ic).copy()
        ir[5:] /= x[12]
        pd = (
            sc.real
            + 0.011033
            + 0.00346 * ir
            + np.where(sc.real >= 0, 0.0066, 0.0044) * ir**2
        )
        sh = branch_power(case.h, v[:19])[0] - hbase
        sl = branch_power(lg, v[19:])[0] - np.array([2, 0, 0])
        sn = np.r_[sh, sl]
        for k, b in enumerate(ATTACH):
            sn[b] -= ss[k]
        out = []
        types = np.r_[case.h.bus[:, 1], case.l.bus[:, 1]]
        for b in range(22):
            if b in (0, 5):
                out.extend([v[b].real - vm[b], v[b].imag])
            elif b in (16, 20):
                out.extend([sn[b].real, sn[b].imag])
            elif types[b] == 2:
                out.extend([sn[b].real, abs(v[b]) ** 2 - vm[b] ** 2])
            else:
                out.extend([sn[b].real, sn[b].imag])
        kcl = ic - ist - 1j * bf * vf
        out.extend(kcl.real)
        out.extend(kcl.imag)
        out.extend(
            [
                ss[0].real + 2,
                ss[0].imag - x[6],
                abs(v[16]) ** 2 - 0.985**2,
                ss[2].real - 1,
                ss[2].imag - x[7],
                ss[3].imag - x[8],
                ss[4].real - x[5],
                ss[4].imag - x[9],
                v[20].real - 1,
                v[20].imag,
                ss[6].imag - x[10],
                pd[3] + pd[5],
                pd[4] + pd[6],
                vd[1] - 1,
            ]
        )
        out.extend(vd * (case.dc_y @ vd) + pd[:3])
        return np.array(out)

    sol = root(
        equations,
        np.r_[initial.real, initial.imag, np.ones(3)],
        options={"xtol": 1e-10, "maxfev": 20000},
    )
    err = float(np.max(abs(equations(sol.x))))
    if err > 1e-8:
        raise RuntimeError(f"Reference failure: {err}")
    v, vf, vc, vd = unpack(sol.x)
    ss = v[ATTACH] * np.conj((vf - v[ATTACH]) / zt)
    trial = ev.evaluate(x)
    delta = float(
        max(
            np.max(abs(v[:19] - trial["vh"])),
            np.max(abs(v[19:] - trial["vl"])),
            np.max(abs(vd - trial["vd"])),
        )
    )
    ds = float(np.max(abs(ss - trial["ss"])))
    return dict(
        reference_residual_pu=err,
        max_voltage_error_pu=delta,
        max_converter_power_error_pu=ds,
        nfev=int(sol.nfev),
        verified=bool(delta < 1e-7 and ds < 1e-7),
    )
