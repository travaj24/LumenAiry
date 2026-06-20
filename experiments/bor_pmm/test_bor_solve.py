"""Validate the BOR-PMM M5 high-level solver (bor_solve) + the achievable GATE 4.

- ENERGY monitor: a structured (ring-grating) stack conserves R+T to the FD
  spurious-mode floor (~1-4%; does NOT improve with N -> the div-conforming SEM
  is the high-accuracy path, also the gate for the production library port).
- GATE 4a (the Cartesian-limit intermediate, rigorous): a uniform interface at
  m!=0 reflects each radial mode with EXACTLY the planar TE/TM Fresnel coefficient
  at that mode's local oblique angle theta = arcsin(gamma/(sqrt(eps) k0)) -- the
  cylindrical->planar correspondence, validated against the closed-form Fresnel
  (independent of both solvers).  This is the load-bearing per-mode physics; the
  full multi-order ring-grating-vs-pmm_efficiency_1d test is SEM-scoped.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from bor_solve import _physical_propagating, build_layer, solve
from zcascade import interface_smatrix


def _uni(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _ring(period, e_lo, e_hi, duty=0.5):
    def f(r):
        e = np.full_like(r, e_lo, dtype=complex)
        e[(r % period) < duty * period] = e_hi
        return e
    return f


def test_structured_stack_energy_floor():
    """A ring-grating stack conserves R+T to the FD spurious-mode floor."""
    m, R, N, k0 = 1, 4.0, 200, 2.0
    layers = [build_layer(m, R, N, _uni(2.0), k0),
              build_layer(m, R, N, _ring(0.8, 2.0, 6.0), k0, thickness=0.5),
              build_layer(m, R, N, _uni(2.0), k0)]
    res = solve(layers, k0)
    assert len(res["inc"]) >= 4                       # multiple physical channels
    assert np.max(np.abs(res["energy"] - 1.0)) < 0.05  # within the ~4% floor


def test_gate4a_planar_fresnel_correspondence():
    """GATE 4a: m!=0 uniform-interface per-mode |S11| == planar TE/TM Fresnel at
    the mode's local oblique angle, to ~1e-3 (measured ~1e-5)."""
    m, R, N, k0 = 1, 5.0, 300, 2.0
    e1, e2 = 4.0, 2.25
    La = build_layer(m, R, N, _uni(e1), k0)
    Lb = build_layer(m, R, N, _uni(e2), k0)
    S11 = interface_smatrix(La["W"], La["V"], Lb["W"], Lb["V"])[0]
    qa = La["q"]
    n_checked = 0
    for j in np.where(_physical_propagating(La))[0]:
        q1 = qa[j].real
        g2 = e1 * k0 ** 2 - q1 ** 2
        if g2 < 0:
            continue
        g = np.sqrt(g2)
        q2 = np.sqrt(e2 * k0 ** 2 - g ** 2 + 0j)
        if q2.imag > 1e-6:
            continue
        rTM = abs((e2 * q1 - e1 * q2) / (e2 * q1 + e1 * q2))
        rTE = abs((q1 - q2) / (q1 + q2))
        s = abs(S11[j, j])
        assert min(abs(s - rTM), abs(s - rTE)) < 1e-3
        n_checked += 1
    assert n_checked >= 5


def test_gate4a_oblique_angles_span():
    """The m=1 modes sample a spread of local oblique angles (not all normal),
    so GATE 4a genuinely exercises the cylindrical-metric curvature."""
    m, R, N, k0 = 1, 5.0, 300, 2.0
    e1 = 4.0
    La = build_layer(m, R, N, _uni(e1), k0)
    qa = La["q"]
    angles = []
    for j in np.where(_physical_propagating(La))[0]:
        g2 = e1 * k0 ** 2 - qa[j].real ** 2
        if g2 > 0:
            angles.append(np.degrees(np.arcsin(np.sqrt(g2) / (np.sqrt(e1) * k0))))
    assert max(angles) - min(angles) > 20.0           # a real oblique spread
