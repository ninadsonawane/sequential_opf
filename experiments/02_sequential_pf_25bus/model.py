"""Independent, explicitly incomplete-data reconstruction of P3 Fig.6.

No loads or impedances are calibrated against the published output tables.
The physical equations are checked by a separate expanded-circuit solution.
"""

import p3_bootstrap as bootstrap  # noqa: F401
from dataclasses import dataclass
import json
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import brentq, root
from pypower.case14 import case14
from grid_utils import Grid, VSC, converter, branch_power, make_ybus
from benchmark import solve_impl, solve_native

BASE = 100.0
ATTACH = np.array([4, 16, 17, 2, 15, 20, 21])  # VSC1..7, zero-based AC bus.


def line(length, kv, freq, distributed=False):
    """P3 Appendix 500kV geometry; extension to 345kV is an assumption."""
    r = 0.01345 if freq == 10 else 0.01540
    z = (r + 1j * 2 * np.pi * freq * 0.0010007) * length
    sh = 1j * 2 * np.pi * freq * 11.5407e-9 * length
    zb = kv**2 / BASE
    if distributed:
        propagation = np.sqrt(z * sh)
        z = z * np.sinh(propagation) / propagation
        sh = 2 * sh * np.tanh(propagation / 2) / propagation
    # Exact-pi shunt has a tiny conductance too. Nominal-pi is the default;
    # distributed diagnostics use their complete complex two-port separately.
    return z / zb, sh * zb


def physical_branch(f, t, length, kv, freq):
    z, sh = line(length, kv, freq)
    return np.array([f, t, z.real, z.imag, sh.imag, 0, 0, 0, 0, 0, 1, -360, 360])


@dataclass
class Case:
    h: Grid
    l: Grid
    dc_y: np.ndarray
    par: VSC
    notes: dict
    initial_grids: tuple


def build_case(load_scale=5.0, internal="physical_345", pole_factor=1.0):
    """Load bundled numerical inputs; do not fetch source programs at runtime.

    The nominal-pi parameters and base dispatch match the original reconstruction.
    Only the disclosed demand scale and DC pole sensitivity can be varied here.
    Both independent initial grids are stored explicitly, before adding the tie
    and before assigning VSC2's AC voltage controller.
    """
    if internal != "physical_345":
        raise ValueError("Only the frozen, disclosed physical_345 case is bundled")
    source = bootstrap.HERE / "data/reconstructed_25bus.json"
    data = json.loads(source.read_text(encoding="utf-8"))

    def grid(name, values):
        arrays = {
            k: np.asarray(values[k], dtype=float) for k in ("bus", "gen", "branch")
        }
        arrays["bus"][:, 2:4] *= load_scale / data["settings"]["load_scale"]
        return Grid(name, **arrays)

    h = grid("HVAC19", data["hvac"])
    l = grid("LFAC3", data["lfac"])
    initial = tuple(
        grid(name, values)
        for name, values in zip(
            ("Stagg5-initial", "IEEE14-initial"), data["initial_grids"]
        )
    )
    yd = np.asarray(data["dc_ybus"], dtype=float) * pole_factor
    notes = dict(data["settings"], load_scale=load_scale, dc_pole_factor=pole_factor)
    return Case(h, l, yd, VSC(**data["converter"]), notes, initial)


def make_args(grid, specified=None, v0=None, tol=1e-9, depth=8):
    y = csr_matrix(make_ybus(grid))
    y.sort_indices()
    vm = grid.bus[:, 7].copy()
    v = vm.astype(complex) if v0 is None else v0.copy()
    refs = np.flatnonzero(grid.bus[:, 1] == 3)
    for i in refs:
        v[i] = vm[i] * np.exp(1j * np.deg2rad(grid.bus[i, 8]))
    return (
        y.indptr.astype(np.int64),
        y.indices.astype(np.int64),
        y.data.astype(np.complex128),
        y.diagonal().astype(np.complex128),
        grid.specified_power() if specified is None else specified,
        v,
        grid.bus[:, 1].astype(np.int64),
        vm,
        tol,
        5000,
        depth,
        1000.0,
    )


def station_rows(v, vh, vl, controls, case):
    p2, p4, p7 = controls
    snh = vh * np.conj(make_ybus(case.h) @ vh) - case.h.specified_power()
    snl = vl * np.conj(make_ybus(case.l) @ vl) - case.l.specified_power()
    ss = np.array(
        [
            -2 - 0.3j,
            complex(p2, snh[16].imag),
            1 + 0.05j,
            complex(p4, -0.3),
            0.4 - 0.1j,
            snl[1],
            complex(p7, 0.1),
        ]
    )
    cs = [converter(v[ATTACH[i]], ss[i], case.par) for i in range(7)]
    return ss, cs


def inverse_station(v, q, required_withdrawal, par):
    fn = lambda p: converter(v, complex(p, q), par)["pdc"] - required_withdrawal
    return brentq(fn, -10, 10, xtol=5e-14)


def solve_dc(yd, pinj, v0=None):
    """Constant injection DC PF; DC24 voltage is exactly 1.0."""

    def f(x):
        vd = np.array([x[0], 1.0, x[1]])
        return (vd * (yd @ vd) - pinj)[[0, 2]]

    start = np.ones(2) if v0 is None else v0[[0, 2]]
    sol = root(f, start, options={"xtol": 1e-11})
    err = np.max(abs(f(sol.x)))
    if err > 1e-9:
        raise RuntimeError(f"DC PF failure: {err}")
    vd = np.array([sol.x[0], 1.0, sol.x[1]])
    return vd, vd * (yd @ vd)


def sequential(
    case, engine="python", depth=8, tol=1e-8, max_outer=100, initial_runner=None
):
    """Fig.5 sequence: initial independent HVAC PF, then coupled AC/DC updates.

    AC4--10 patch joins the two same-frequency areas. Each area keeps its fixed
    slack phasor; generator outputs balance those areas. LF has its own phase.
    """
    fun = solve_native if engine == "numba" else solve_impl
    init_args = [make_args(g, tol=tol / 10, depth=depth) for g in case.initial_grids]
    if initial_runner is None:
        initial = [fun(*a) for a in init_args]
    else:
        initial = initial_runner()
    for output in initial:
        if output[2] > tol / 10:
            raise RuntimeError(f"Initial independent AC solve failed: {output[2]}")
    vh = np.r_[initial[0][0], initial[1][0]]
    # New converter-controlled PV17 target.
    vh[16] *= 0.985 / abs(vh[16])
    vl = np.ones(3, complex)
    vd = np.ones(3)
    powers = np.array([1.0, 1.6, -0.4])  # Eq22, ignoring losses for initialization.
    trace = []
    total_sweeps = sum(o[1] for o in initial)
    hbase = case.h.specified_power()
    lbase = case.l.specified_power()
    yh = make_ybus(case.h)
    yl = make_ybus(case.l)
    for it in range(1, max_outer + 1):
        sh = hbase.copy()
        sl = lbase.copy()
        sh[[4, 16, 17, 2, 15]] += [
            -2 - 0.3j,
            powers[0] + 0j,
            1 + 0.05j,
            powers[1] - 0.3j,
            0.4 - 0.1j,
        ]
        sl[2] += powers[2] + 0.1j
        ah = make_args(case.h, sh, vh, tol / 10, depth)
        al = make_args(case.l, sl, vl, tol / 10, depth)
        rh = fun(*ah)
        rl = fun(*al)
        if max(rh[2], rl[2]) > tol / 10:
            raise RuntimeError(f"Coupled AC failed at outer {it}: {rh[2]}, {rl[2]}")
        vh, vl = rh[0], rl[0]
        v = np.r_[vh, vl]
        total_sweeps += rh[1] + rl[1]
        ss, cs = station_rows(v, vh, vl, powers, case)
        pd = -np.array([c["pdc"] for c in cs[:3]])
        vd, dp = solve_dc(case.dc_y, pd, vd)
        balance = np.array(
            [
                cs[1]["pdc"] + dp[1],
                cs[3]["pdc"] + cs[5]["pdc"],
                cs[4]["pdc"] + cs[6]["pdc"],
            ]
        )
        newpowers = np.array(
            [
                inverse_station(v[16], ss[1].imag, -dp[1], case.par),
                inverse_station(v[2], -0.3, -cs[5]["pdc"], case.par),
                inverse_station(v[21], 0.1, -cs[4]["pdc"], case.par),
            ]
        )
        change = float(np.max(abs(newpowers - powers)))
        trace.append(
            dict(
                iteration=it,
                dc_interface_mismatch_pu=float(max(abs(balance))),
                p_update_pu=change,
                hvac_ac_mismatch_pu=float(rh[2]),
                lfac_ac_mismatch_pu=float(rl[2]),
                hvac_sweeps=int(rh[1]),
                lfac_sweeps=int(rl[1]),
                p2_mw=float(powers[0] * BASE),
                p4_mw=float(powers[1] * BASE),
                p7_mw=float(powers[2] * BASE),
            )
        )
        if max(change, max(abs(balance))) < tol:
            return dict(
                vh=vh,
                vl=vl,
                vd=vd,
                ss=ss,
                converters=cs,
                trace=trace,
                total_sweeps=int(total_sweeps),
                outer_iterations=it,
                initial_iterations=[int(o[1]) for o in initial],
            )
        powers = newpowers
    raise RuntimeError(f"Outer iteration failed: {trace[-1]}")


def initial_task(task):
    args, engine = task
    return (solve_native if engine == "numba" else solve_impl)(*args)


def simultaneous_reference(case):
    """75 real unknowns: 22 bus + 14 converter internal phasors + 3 DC V.

    Explicit branch-current assembly, filter KCL, PQ/PV/slack controls and DC
    nodal balances. Independent flat start; no sequential output as initial guess.
    """
    h, l, par = case.h, case.l, case.par
    vm = np.r_[h.bus[:, 7], l.bus[:, 7]]
    start = np.r_[
        vm.astype(complex), vm[ATTACH].astype(complex), vm[ATTACH].astype(complex)
    ]
    count = len(start)

    def unpack(x):
        z = x[:count] + 1j * x[count : 2 * count]
        return z[:22], z[22:29], z[29:], x[2 * count :]

    def residual(x):
        v, vf, vc, vd = unpack(x)
        ist = (vf - v[ATTACH]) / par.zt
        ic = (vc - vf) / par.zc
        ss = v[ATTACH] * np.conj(ist)
        sc = vc * np.conj(ic)
        withdrawal = (
            sc.real
            + par.a0
            + par.a1 * abs(ic)
            + np.where(sc.real >= 0, par.a2_inv, par.a2_rect) * abs(ic) ** 2
        )
        sn = np.r_[
            branch_power(h, v[:19])[0] - h.specified_power(),
            branch_power(l, v[19:])[0] - l.specified_power(),
        ]
        for k, i in enumerate(ATTACH):
            sn[i] -= ss[k]
        out = []
        types = np.r_[h.bus[:, 1], l.bus[:, 1]]
        for i in range(22):
            if i in (0, 5):  # Generator slacks.
                out.extend([v[i].real - vm[i], v[i].imag])
            elif i in (
                16,
                20,
            ):  # Converter-controlled AC voltage/slack, controls below.
                out.extend([sn[i].real, sn[i].imag])
            elif types[i] == 2:
                out.extend([sn[i].real, abs(v[i]) ** 2 - vm[i] ** 2])
            else:
                out.extend([sn[i].real, sn[i].imag])
        kcl = ic - ist - 1j * par.bf * vf
        out.extend(kcl.real)
        out.extend(kcl.imag)
        # 14 station control/balance constraints.
        out.extend(
            [
                ss[0].real + 2,
                ss[0].imag + 0.3,
                abs(v[16]) ** 2 - 0.985**2,
                ss[2].real - 1,
                ss[2].imag - 0.05,
                ss[3].imag + 0.3,
                ss[4].real - 0.4,
                ss[4].imag + 0.1,
                v[20].real - 1,
                v[20].imag,
                ss[6].imag - 0.1,
                withdrawal[3] + withdrawal[5],
                withdrawal[4] + withdrawal[6],
                vd[1] - 1,
            ]
        )
        out.extend(vd * (case.dc_y @ vd) + withdrawal[:3])
        return np.array(out)

    x0 = np.r_[start.real, start.imag, np.ones(3)]
    assert len(residual(x0)) == len(x0) == 75
    sol = root(residual, x0, options={"xtol": 1e-10, "maxfev": 20000})
    err = float(np.max(abs(residual(sol.x))))
    if err > 1e-8:
        raise RuntimeError(f"Expanded-circuit reference failed: {sol.message}; {err}")
    v, vf, vc, vd = unpack(sol.x)
    ss = v[ATTACH] * np.conj((vf - v[ATTACH]) / par.zt)
    return dict(
        vh=v[:19],
        vl=v[19:],
        vd=vd,
        vf=vf,
        vc=vc,
        ss=ss,
        residual=err,
        nfev=int(sol.nfev),
        success=bool(sol.success),
    )


def validate(case, result, reference):
    vh, vl, vd, ss = [result[k] for k in ("vh", "vl", "vd", "ss")]
    # Outputs here are nodal complex POWER, assembled from branch currents.
    nodal_h, flows_h = branch_power(case.h, vh)
    nodal_l, flows_l = branch_power(case.l, vl)
    conv_h = np.zeros(19, complex)
    conv_l = np.zeros(3, complex)
    for k, i in enumerate(ATTACH):
        if i < 19:
            conv_h[i] += ss[k]
        else:
            conv_l[i - 19] += ss[k]
    mismatch_h = nodal_h - case.h.specified_power() - conv_h
    mismatch_l = nodal_l - case.l.specified_power() - conv_l
    err = 0.0
    for grid, d in ((case.h, mismatch_h), (case.l, mismatch_l)):
        for i in range(grid.n):
            if grid.bus[i, 1] == 3:
                continue
            err = max(err, abs(d[i].real))
            if grid.bus[i, 1] == 1:
                err = max(err, abs(d[i].imag))
    needed_h = nodal_h + (case.h.bus[:, 2] + 1j * case.h.bus[:, 3]) / BASE - conv_h
    pg = sum(needed_h[int(g[0]) - 1].real for g in case.h.gen) + 2
    demand = sum(case.h.bus[:, 2]) / BASE
    lh = sum((f[2] + f[3]).real for f in flows_h)
    ll = sum((f[2] + f[3]).real for f in flows_l)
    ld = float(vd @ (case.dc_y @ vd))
    lc = sum(c["station_loss"] for c in result["converters"])
    dcres = vd * (case.dc_y @ vd) + np.array(
        [c["pdc"] for c in result["converters"][:3]]
    )
    links = [
        result["converters"][i]["pdc"] + result["converters"][j]["pdc"]
        for i, j in [(3, 5), (4, 6)]
    ]
    checks = dict(
        ac_max_mismatch_pu=float(err),
        dc_max_mismatch_pu=float(max(abs(dcres))),
        b2b_max_imbalance_pu=float(max(abs(np.array(links)))),
        global_power_balance_error_mw=float((pg - demand - lh - ll - ld - lc) * BASE),
        independent_max_ac_phasor_error_pu=float(
            max(np.max(abs(vh - reference["vh"])), np.max(abs(vl - reference["vl"])))
        ),
        independent_max_dc_voltage_error_pu=float(max(abs(vd - reference["vd"]))),
        independent_max_converter_power_error_pu=float(max(abs(ss - reference["ss"]))),
        independent_equation_residual_pu=reference["residual"],
    )
    checks["numerically_verified"] = bool(
        err < 1e-8
        and max(abs(dcres)) < 1e-8
        and max(abs(np.array(links))) < 1e-8
        and abs(checks["global_power_balance_error_mw"]) < 1e-5
        and checks["independent_max_ac_phasor_error_pu"] < 1e-7
        and checks["independent_max_dc_voltage_error_pu"] < 1e-7
        and checks["independent_max_converter_power_error_pu"] < 1e-7
    )
    limits = []
    for k, c in enumerate(result["converters"]):
        limits.append(
            dict(
                station=k + 1,
                ic_pu=float(abs(c["ic"])),
                vc_pu=float(abs(c["vc"])),
                current_limit_ok=bool(abs(c["ic"]) <= case.par.imax),
                converter_voltage_limit_ok=bool(
                    case.par.vcmin <= abs(c["vc"]) <= case.par.vcmax
                ),
            )
        )
    return dict(
        checks=checks,
        limits=limits,
        power=dict(
            generation_mw=float(pg * BASE),
            demand_mw=float(demand * BASE),
            hvac_line_loss_mw=float(lh * BASE),
            lfac_line_loss_mw=float(ll * BASE),
            dc_line_loss_mw=float(ld * BASE),
            station_loss_mw=float(lc * BASE),
            total_loss_mw=float((lh + ll + ld + lc) * BASE),
        ),
        flows_h=flows_h,
        flows_l=flows_l,
    )
