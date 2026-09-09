"""Probe 18 -- ADVERSARIAL: are the NEW Eq.40 / Eq.44 / Eq.25 terms actually
LOAD-BEARING, or does the tensor path happen to pass its gates without them?

Each arm monkeypatches ONE new contribution to zero (at the class level, no
source edit) and re-measures the G3 Berreman residual, which is the tightest
oracle in the build (1e-14 when everything is present).  A term that is not
load-bearing would leave that residual unchanged -- the "right conclusion,
wrong mechanism" failure this probe exists to rule out.
"""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

WL, P, DEP = 1.0e-6, 0.40e-6, 0.55e-6
NS, NC = 1.5, 1.0
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)

_EW = TS.Granet2DTransverseE._eps_weighted
_ED = TS.Granet2DTransverseE._eps_dir
_RM = TS._region_modes


def residual(t33, M=7, theta=25 * np.pi / 180, phi=40 * np.pi / 180):
    cell = np.empty((2, 2, 3, 3), dtype=complex)
    cell[:] = t33
    _o, R, T, J = TS.pmm_jones_2d_staggered(P, P, cell, NS, NC, DEP, WL,
                                            degree=M, n_orders=2,
                                            theta=theta, phi=phi)
    Rb, Tb, jr, _jt = berreman_jones_1d([(t33, DEP)], NS, NC, WL, angle=theta,
                                        phi=phi)
    return max(float(np.max(np.abs(R.sum(axis=1) - Rb))),
               float(np.max(np.abs(T.sum(axis=1) - Tb))),
               float(np.max(np.abs(J - jr))))


def _no_mixed_mass(self, refx_pair, refy_pair, wmap=None):
    """Drop the Eq.40 MIXED masses (the two blocks whose 1-D set pairs are
    UNLIKE: <B|Btil> and <Btil|B>)."""
    out = _EW(self, refx_pair, refy_pair, wmap)
    if refx_pair[2] is not refx_pair[3]:
        return np.zeros_like(out)
    return out


def _no_second_kzt(self, bx, lx, opx, rx, by, ly, opy, ry, wmap=None):
    """Drop the SECOND term of each Eq.44 K_zt column (the one whose
    derivative sits on the OTHER axis than the shipped isotropic term)."""
    out = _ED(self, bx, lx, opx, rx, by, ly, opy, ry, wmap)
    if (opx, rx, opy, ry) in (("m", "B", "dL", "Btilde"),
                              ("dL", "Btilde", "m", "B")):
        return np.zeros_like(out)
    return out


def _no_lhh_mixed(solver):
    """Drop the Eq.25 H-partner's mixed blocks only (Lmat keeps them)."""
    saved = solver.Et_offdiag
    solver.Et_offdiag = None
    try:
        return _RM(solver)
    finally:
        solver.Et_offdiag = saved


ARMS = [
    ("ALL TERMS PRESENT (reference)", None),
    ("Eq.40 mixed masses -> 0", "mass"),
    ("Eq.44 second K_zt term -> 0", "kzt"),
    ("Eq.25 Lhh mixed blocks -> 0", "lhh"),
]

print(f"{'arm':34s} {'LC slab':>12s} {'gyrotropic slab':>18s}")
for label, mode in ARMS:
    TS.Granet2DTransverseE._eps_weighted = (_no_mixed_mass if mode == "mass"
                                            else _EW)
    TS.Granet2DTransverseE._eps_dir = (_no_second_kzt if mode == "kzt"
                                       else _ED)
    TS._region_modes = _no_lhh_mixed if mode == "lhh" else _RM
    # stack2d_pure imported _region_modes by value -- patch its binding too
    import lumenairy.elements.pmm.stack2d_pure as SP
    SP._region_modes = TS._region_modes
    try:
        a, b = residual(LC), residual(GYRO)
        print(f"{label:34s} {a:12.3e} {b:18.3e}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"{label:34s} RAISED {type(exc).__name__}: {exc}")
    finally:
        TS.Granet2DTransverseE._eps_weighted = _EW
        TS.Granet2DTransverseE._eps_dir = _ED
        TS._region_modes = _RM
        SP._region_modes = _RM
