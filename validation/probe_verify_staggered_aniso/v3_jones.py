"""V3 -- CROSS-ENGINE JONES verification (magnitudes AND phases) for
``pmm_jones_2d_staggered``.

Every efficiency gate in the build is blind to a global conjugation of the
reflection Jones (|J|^2 is unchanged) and to a TRANSPOSE of it whenever the
medium is reciprocal-symmetric (an LC tensor gives J01 = J10).  This probe
attacks exactly those two blind spots:

  A. complex Jones vs ``berreman_jones_1d`` (uniform tensor slab, exact
     oracle, documented convention "columns = incident lab [Ex; Ey]"),
     normal / oblique / conical, with the CONJUGATED arm measured as the
     fail-before.
  B. complex Jones vs ``pmm_jones_2d`` (hybrid; solves in an internal
     conjugated gauge and conjugates back) and ``rcwa_jones_2d`` on a
     PATTERNED in-plane anisotropic cell, normal and oblique.
  C. row/column convention: on a GYROTROPIC slab reciprocity forces
     J01 = -J10, so the TRANSPOSED arm is a genuine fail-before (on an LC
     tensor J01 = J10 and a transpose is invisible -- measured, to show why
     the gyrotropic fixture is the one that discriminates).
  D. |jones|^2 <-> efficiency consistency: order-0 reflected power rebuilt
     from the Jones (including the Ez component forced by k.E = 0) against
     the engine's own R00.
"""
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath("C:/tmp/lum_aniso")
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

from lumenairy.elements.berreman import berreman_jones_1d  # noqa: E402
from lumenairy.elements.pmm import pmm_jones_2d  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

LC = uniaxial_tensor(1.45, 1.75, np.pi / 2, phi=0.90)
GY = np.array([[2.60, 0.35j, 0.0], [-0.35j, 2.60, 0.0],
               [0.0, 0.0, 2.40]], dtype=complex)
ISO = 5.0 * np.eye(3, dtype=complex)
DIAG = np.diag([2.10, 3.40, 2.70]).astype(complex)      # e12 = e21 = 0


def uni(t, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = t
    return c


def cell(host, pillar, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = pillar
    return c


def jrec(J):
    J = np.asarray(J)
    return dict(abs=np.abs(J).ravel().tolist(),
                deg=np.degrees(np.angle(J)).ravel().tolist())


OUT = {}

# ---------------------------------------------------------------- A + C
P, WL, DEP, NSUB = 0.33e-6, 0.90e-6, 0.62e-6, 1.6
A = {}
for name, t33 in (("lc", LC), ("gyro", GY), ("diag", DIAG)):
    for th, ph in ((0.0, 0.0), (0.30, 0.0), (0.30, 0.75)):
        _o, R, T, J = pmm_jones_2d_staggered(P, P, uni(t33), NSUB, 1.0, DEP,
                                             WL, degree=7, n_orders=2,
                                             theta=th, phi=ph)
        Rb, Tb, jr, _jt = berreman_jones_1d([(t33, DEP)], NSUB, 1.0, WL,
                                            angle=th, phi=ph)
        k = f"{name}_th{th:.2f}_ph{ph:.2f}"
        A[k] = dict(
            jones_stag=jrec(J), jones_berreman=jrec(jr),
            d_complex=float(np.max(np.abs(J - jr))),
            d_abs=float(np.max(np.abs(np.abs(J) - np.abs(jr)))),
            d_phase_deg=float(np.max(np.abs(np.degrees(
                np.angle(J / np.where(np.abs(jr) > 1e-12, jr, 1.0)))))),
            FAILBEFORE_conjugated=float(np.max(np.abs(np.conj(J) - jr))),
            FAILBEFORE_transposed=float(np.max(np.abs(J.T - jr))),
            offdiag_ratio_J01_over_J10=complex(
                J[0, 1] / J[1, 0]).__repr__() if abs(J[1, 0]) > 1e-14 else None,
            R=R.sum(axis=1).tolist(), R_berreman=np.asarray(Rb).tolist(),
            dR=float(np.max(np.abs(R.sum(axis=1) - Rb))),
        )
OUT["A_berreman_uniform"] = A

# ---------------------------------------------------------------- B
P2, WL2, DEP2, NS2 = 0.62e-6, 0.50e-6, 0.31e-6, 1.45


def big(c, up):
    n = 2 * up
    out = np.empty((n, n, 3, 3), dtype=complex)
    out[:] = c[1, 1]
    out[:up, :up] = c[0, 0]
    out[:up, up:] = c[0, 1]
    out[up:, :up] = c[1, 0]
    return out


B = {}
for th, ph in ((0.0, 0.0), (0.30, 0.70)):
    c = cell(LC, ISO)
    _o, R, T, Js = pmm_jones_2d_staggered(P2, P2, c, NS2, 1.0, DEP2, WL2,
                                          degree=7, n_orders=5, theta=th,
                                          phi=ph)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _oh, Rh, Th, Jh = pmm_jones_2d(P2, P2, c, NS2, 1.0, DEP2, WL2,
                                       degree=11, n_orders=13, theta=th,
                                       phi=ph)
    _or, Rr, Tr, Jr = rcwa_jones_2d(P2, P2, big(c, 32), NS2, 1.0, DEP2, WL2,
                                    n_orders_x=13, n_orders_y=13, theta=th,
                                    phi=ph)
    k = f"patterned_th{th:.2f}_ph{ph:.2f}"
    B[k] = dict(
        jones_stag=jrec(Js), jones_hybrid=jrec(Jh), jones_rcwa=jrec(Jr),
        d_stag_hybrid=float(np.max(np.abs(Js - Jh))),
        d_stag_rcwa=float(np.max(np.abs(Js - Jr))),
        d_hybrid_rcwa=float(np.max(np.abs(Jh - Jr))),
        d_phase_stag_hybrid_deg=float(np.max(np.abs(
            np.degrees(np.angle(Js / Jh))))),
        d_phase_stag_rcwa_deg=float(np.max(np.abs(
            np.degrees(np.angle(Js / Jr))))),
        FAILBEFORE_conj_vs_hybrid=float(np.max(np.abs(np.conj(Js) - Jh))),
        FAILBEFORE_conj_vs_rcwa=float(np.max(np.abs(np.conj(Js) - Jr))),
        FAILBEFORE_transpose_vs_hybrid=float(np.max(np.abs(Js.T - Jh))),
    )
OUT["B_patterned_cross_engine"] = B

# ---------------------------------------------------------------- D
D = {}
for th, ph in ((0.0, 0.0), (0.30, 0.70)):
    c = cell(LC, ISO)
    o, R, T, J = pmm_jones_2d_staggered(P2, P2, c, NS2, 1.0, DEP2, WL2,
                                        degree=7, n_orders=5, theta=th,
                                        phi=ph)
    i0 = int(np.where((np.asarray(o)[:, 0] == 0)
                      & (np.asarray(o)[:, 1] == 0))[0][0])
    kx = np.sin(th) * np.cos(ph)          # / k0, superstrate n = 1
    ky = np.sin(th) * np.sin(ph)
    kz = np.sqrt(1.0 - kx ** 2 - ky ** 2)
    pred = []
    for col in (0, 1):
        ex, ey = J[0, col], J[1, col]
        ez = (kx * ex + ky * ey) / kz     # k_r = (kx, ky, -kz), k.E = 0
        pred.append(float(abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2))
    D[f"th{th:.2f}_ph{ph:.2f}"] = dict(
        R00_engine=[float(R[0][i0]), float(R[1][i0])],
        R00_from_jones=pred,
        d=float(np.max(np.abs(np.array(pred)
                              - np.array([R[0][i0], R[1][i0]])))),
        R00_from_jones_no_ez=[float(abs(J[0, c]) ** 2 + abs(J[1, c]) ** 2)
                              for c in (0, 1)])
OUT["D_jones_power_consistency"] = D

fn = "C:/tmp/lum_aniso/validation/probe_verify_staggered_aniso/out_v3_jones.json"
json.dump(OUT, open(fn, "w"), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
print("written", fn)
