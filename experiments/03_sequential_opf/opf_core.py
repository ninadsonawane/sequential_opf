"""Compiled/Python identical sequential PF for P1-style reduced-space OPF.

Balanced positive sequence, 25 buses, common 100MVA. No learning/inference.
"""

import opf_bootstrap
import numpy as np
from numba import njit
from numba.extending import register_jitable
from benchmark import solve_impl

gs_anderson = register_jitable(solve_impl)
ATTACH = np.array([4, 16, 17, 2, 15, 20, 21], dtype=np.int64)
PG_BUS = np.array([1, 6, 7, 10, 12], dtype=np.int64)
GVM_BUS = np.array([0, 1, 5, 6, 7, 10, 12], dtype=np.int64)


@register_jitable
def parameters(frequency_ratio, voltage_ratio):
    zt = np.full(7, 0.0015 + 1j * 0.1121, np.complex128)
    zc = np.full(7, 0.0001 + 1j * 0.16428, np.complex128)
    bf = np.full(7, 0.0887, np.float64)
    scale = voltage_ratio**2
    for k in (5, 6):
        zt[k] = (0.0015 + 1j * 0.1121 * frequency_ratio) / scale
        zc[k] = (0.0001 + 1j * 0.16428 * frequency_ratio) / scale
        bf[k] = 0.0887 * frequency_ratio * scale
    return zt, zc, bf


@register_jitable
def station(v, s, k, zt, zc, bf, voltage_ratio):
    ist = (s / v).conjugate()
    vf = v + zt[k] * ist
    ic = ist + 1j * bf[k] * vf
    vc = vf + zc[k] * ic
    sc = vc * ic.conjugate()
    current = abs(ic) / (voltage_ratio if k >= 5 else 1.0)
    a2 = 0.0066 if sc.real >= 0 else 0.0044
    loss = 0.011033 + 0.00346 * current + a2 * current**2
    joule = zt[k].real * abs(ist) ** 2 + zc[k].real * abs(ic) ** 2
    return sc.real + loss, ic, vc, sc, loss, joule


@register_jitable
def inverse_station(v, q, required, k, zt, zc, bf, vr):
    p = required
    for _ in range(20):
        err = station(v, complex(p, q), k, zt, zc, bf, vr)[0] - required
        if abs(err) < 1e-12:
            break
        step = 1e-5
        deriv = (
            station(v, complex(p + step, q), k, zt, zc, bf, vr)[0]
            - station(v, complex(p - step, q), k, zt, zc, bf, vr)[0]
        ) / (2 * step)
        p -= err / deriv
    return p


@register_jitable
def lf_admittance(fr, vr, resistance_model):
    freq = 10 * fr
    if resistance_model == 0:
        r = 0.01345  # P1-style constant-R approximation, referenced to P3 10Hz.
    elif freq <= 10:
        r = 0.01325 + (0.01345 - 0.01325) * freq / 10
    else:
        r = 0.01345 + (0.01540 - 0.01345) * (freq - 10) / 50
    zb = (500 * vr) ** 2 / 100
    z = (r + 1j * 2 * np.pi * freq * 0.0010007) * 1000 / zb
    b = 2 * np.pi * freq * 11.5407e-9 * 1000 * zb
    y = 1 / z
    data = np.array(
        [2 * y + 1j * b, -y, -y, -y, y + 1j * b / 2, -y, y + 1j * b / 2],
        dtype=np.complex128,
    )
    diagonal = np.array([data[0], data[4], data[6]], dtype=np.complex128)
    return data, diagonal, z, b


@register_jitable
def dc_solve(yd, pinj, vd):
    for _ in range(20):
        current = yd @ vd
        mismatch = vd * current - pinj
        if max(abs(mismatch[0]), abs(mismatch[2])) < 1e-11:
            break
        j00 = current[0] + vd[0] * yd[0, 0]
        j02 = vd[0] * yd[0, 2]
        j20 = vd[2] * yd[2, 0]
        j22 = current[2] + vd[2] * yd[2, 2]
        det = j00 * j22 - j02 * j20
        d0 = (mismatch[0] * j22 - j02 * mismatch[2]) / det
        d2 = (j00 * mismatch[2] - mismatch[0] * j20) / det
        vd[0] -= d0
        vd[2] -= d2
    return vd, vd * (yd @ vd)


def core_impl(
    hptr,
    hcols,
    hdata,
    hdiag,
    hkind,
    hvm,
    hdemand,
    yd,
    x,
    profile,
    resistance_model,
    inner_tol,
    outer_tol,
):
    """x=[five Pg, P5, Q1,Q3,Q4,Q5,Q7, f/10, rated_kV/500, seven generator Vm].

    P1=-200MW and P3=100MW are retained transfer obligations. LF generator
    is a must-take 200MW forecast. Slack outputs and P2/P4/P7 are dependent.
    """
    fr = x[11]
    vr = x[12]
    hvm = hvm.copy()
    for k in range(7):
        hvm[GVM_BUS[k]] = x[13 + k]
    zt, zc, bf = parameters(fr, vr)
    lp = np.array([0, 3, 5, 7], dtype=np.int64)
    lc = np.array([0, 1, 2, 0, 1, 0, 2], dtype=np.int64)
    ld, ldiag, zline, bline = lf_admittance(fr, vr, resistance_model)
    lkind = np.array([2, 3, 1], dtype=np.int64)
    lvm = np.ones(3)
    lbase = np.array([2 + 0j, 0 + 0j, 0 + 0j], dtype=np.complex128)
    hbase = -hdemand * profile
    for k in range(5):
        hbase[PG_BUS[k]] += x[k]
    vh = hvm.astype(np.complex128)
    vl = np.ones(3, dtype=np.complex128)
    vd = np.ones(3)
    ss = np.zeros(7, dtype=np.complex128)
    powers = np.array([1.0, 2 - x[5], -x[5]])
    currents = np.zeros(7, dtype=np.complex128)
    cvolts = currents.copy()
    sc = currents.copy()
    losses = np.zeros(7)
    joule = np.zeros(7)
    withdraw = np.zeros(7)
    sweeps = 0
    ac_error = 1e100
    interface_error = 1e100
    converged = False
    for iteration in range(1, 61):
        sh = hbase.copy()
        sl = lbase.copy()
        sh[4] += complex(-2, x[6])
        sh[16] += powers[0]
        sh[17] += complex(1, x[7])
        sh[2] += complex(powers[1], x[8])
        sh[15] += complex(x[5], x[9])
        sl[2] += complex(powers[2], x[10])
        rh = gs_anderson(
            hptr, hcols, hdata, hdiag, sh, vh, hkind, hvm, inner_tol, 600, 8, 1000.0
        )
        rl = gs_anderson(
            lp, lc, ld, ldiag, sl, vl, lkind, lvm, inner_tol, 600, 8, 1000.0
        )
        vh = rh[0]
        vl = rl[0]
        sweeps += rh[1] + rl[1]
        ac_error = max(rh[2], rl[2])
        if not np.isfinite(ac_error) or ac_error > inner_tol * 1.01:
            break
        # Q at AC17 and complex slack power at LF21 are recovered by nodal KCL.
        ih = 0j
        for j in range(hptr[16], hptr[17]):
            ih += hdata[j] * vh[hcols[j]]
        q2 = (vh[16] * ih.conjugate() - hbase[16]).imag
        il = ld[3] * vl[0] + ld[4] * vl[1]
        ss[0] = complex(-2, x[6])
        ss[1] = complex(powers[0], q2)
        ss[2] = complex(1, x[7])
        ss[3] = complex(powers[1], x[8])
        ss[4] = complex(x[5], x[9])
        ss[5] = vl[1] * il.conjugate()
        ss[6] = complex(powers[2], x[10])
        volts = np.concatenate((vh, vl))
        for k in range(7):
            c = station(volts[ATTACH[k]], ss[k], k, zt, zc, bf, vr)
            withdraw[k] = c[0]
            currents[k] = c[1]
            cvolts[k] = c[2]
            sc[k] = c[3]
            losses[k] = c[4]
            joule[k] = c[5]
        vd, dcnet = dc_solve(yd, -withdraw[:3], vd)
        dcerr = withdraw[1] + dcnet[1]
        balance1 = withdraw[3] + withdraw[5]
        balance2 = withdraw[4] + withdraw[6]
        updated = np.array(
            [
                inverse_station(vh[16], q2, -dcnet[1], 1, zt, zc, bf, vr),
                inverse_station(vh[2], x[8], -withdraw[5], 3, zt, zc, bf, vr),
                inverse_station(vl[2], x[10], -withdraw[4], 6, zt, zc, bf, vr),
            ]
        )
        change = np.max(np.abs(updated - powers))
        interface_error = max(abs(dcerr), abs(balance1), abs(balance2), change)
        if interface_error < outer_tol:
            converged = True
            break
        powers = updated
    return (
        vh,
        vl,
        vd,
        ss,
        currents,
        cvolts,
        sc,
        losses,
        joule,
        withdraw,
        ac_error,
        interface_error,
        iteration,
        sweeps,
        converged,
    )


core_native = njit(cache=True, nogil=True, fastmath=False)(core_impl)
