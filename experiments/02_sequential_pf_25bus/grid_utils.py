"""Shared network and converter circuit primitives for the 25-bus experiment.

This is a documented subset reproduction, not the authors' source code. All
powers are positive INTO an AC grid; all per-unit quantities use 100 MVA.
See README.md for the paper mapping, model choices, and notation corrections.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import numpy as np
from scipy.optimize import brentq, root

ROOT = Path(__file__).resolve().parent


def read_matpower(path):
    text = Path(path).read_text()
    case = {
        "version": "2",
        "baseMVA": float(re.search(r"mpc.baseMVA\s*=\s*([\d.]+)", text)[1]),
    }
    for name in ("bus", "gen", "branch", "gencost"):
        body = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", text, re.S)[1]
        body = re.sub(r"%[^\n]*", "", body)
        case[name] = np.array(
            [[float(x) for x in row.split()] for row in body.split(";") if row.strip()]
        )
    return case


@dataclass
class Grid:
    name: str
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    base: float = 100.0

    @property
    def n(self):
        return len(self.bus)

    @property
    def ref(self):
        return int(np.flatnonzero(self.bus[:, 1] == 3)[0])

    @property
    def pv(self):
        return np.flatnonzero(self.bus[:, 1] == 2)

    @property
    def pq(self):
        return np.flatnonzero(self.bus[:, 1] == 1)

    def specified_power(self, conv=None):
        s = -(self.bus[:, 2] + 1j * self.bus[:, 3]) / self.base
        for g in self.gen:
            if g[7]:
                s[int(g[0]) - 1] += (g[1] + 1j * g[2]) / self.base
        if conv is not None:
            s = s + conv
        return s

    def initial_voltage(self):
        v = np.ones(self.n, dtype=complex)
        for g in self.gen:
            if g[7]:
                v[int(g[0]) - 1] = g[5]
        v[self.ref] = self.bus[self.ref, 7] * np.exp(
            1j * np.deg2rad(self.bus[self.ref, 8])
        )
        return v


def branch_terms(row):
    _, _, r, x, b = row[:5]
    y = 1 / (r + 1j * x)
    tap = (row[8] if row[8] else 1) * np.exp(1j * np.deg2rad(row[9]))
    return (y + 1j * b / 2) / abs(tap) ** 2, -y / np.conj(tap), -y / tap, y + 1j * b / 2


def make_ybus(grid):
    y = np.diag((grid.bus[:, 4] + 1j * grid.bus[:, 5]) / grid.base).astype(complex)
    for row in grid.branch:
        if not row[10]:
            continue
        i, j = int(row[0]) - 1, int(row[1]) - 1
        a, b, c, d = branch_terms(row)
        y[i, i] += a
        y[i, j] += b
        y[j, i] += c
        y[j, j] += d
    return y


def branch_power(grid, v):
    """Independent explicit branch-current assembly, also returns terminal powers."""
    current = (grid.bus[:, 4] + 1j * grid.bus[:, 5]) / grid.base * v
    flows = []
    for row in grid.branch:
        if not row[10]:
            continue
        i, j = int(row[0]) - 1, int(row[1]) - 1
        z = complex(row[2], row[3])
        b = row[4]
        tap = (row[8] or 1.0) * np.exp(1j * np.deg2rad(row[9]))
        vf = v[i] / tap
        iseries = (vf - v[j]) / z
        i_from = (iseries + 1j * b / 2 * vf) / np.conj(tap)
        i_to = -iseries + 1j * b / 2 * v[j]
        current[i] += i_from
        current[j] += i_to
        flows.append((i + 1, j + 1, v[i] * np.conj(i_from), v[j] * np.conj(i_to)))
    return v * np.conj(current), flows


@dataclass
class VSC:
    rt: float = 0.0015
    xt: float = 0.1121
    bf: float = 0.0887
    rc: float = 0.0001
    xc: float = 0.16428
    a0: float = 0.011033
    a1: float = 0.00346
    a2_rect: float = 0.0044
    a2_inv: float = 0.0066
    imax: float = 1.05
    vcmin: float = 0.85
    vcmax: float = 1.20

    @property
    def zt(self):
        return complex(self.rt, self.xt)

    @property
    def zc(self):
        return complex(self.rc, self.xc)


def converter(v, s, par):
    """Fig.4 KCL/KVL; Pdc is withdrawal FROM the shared DC link."""
    is_ = np.conj(s / v)
    vf = v + par.zt * is_
    ic = is_ + 1j * par.bf * vf
    vc = vf + par.zc * ic
    sc = vc * np.conj(ic)
    a2 = par.a2_inv if sc.real >= 0 else par.a2_rect
    psw = par.a0 + par.a1 * abs(ic) + a2 * abs(ic) ** 2
    pjoule = par.rt * abs(is_) ** 2 + par.rc * abs(ic) ** 2
    return {
        "s": s,
        "is": is_,
        "vf": vf,
        "ic": ic,
        "vc": vc,
        "sc": sc,
        "psw": float(psw),
        "p_joule": float(pjoule),
        "pdc": float(sc.real + psw),
        "mode": "inverter" if sc.real >= 0 else "rectifier",
        "station_loss": float(pjoule + psw),
        "energy_identity_error": float(sc.real - s.real - pjoule),
    }
