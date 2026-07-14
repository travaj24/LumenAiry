"""Validate the BOR-PMM M5 high-level solver (bor_solve) + the achievable GATE 4.

- ENERGY monitor, staggered default: ``build_layer`` now defaults to the Yee
  div-conforming basis (the production ``BORStack`` discretization), so the
  structured (ring-grating) stack conserves R+T to machine precision.  The
  legacy NODAL basis (``basis='nodal'``) is only floor-accurate (~1-4%; does
  NOT improve with N) at SMALL cells and blows up entirely at large ones (its
  zero-flux spurious modes orient by the sign of noise, making the interface
  transmission block singular -- see the bor_solve module docstring); its gate
  here pins the documented small-cell floor.
- GATE 4a (the Cartesian-limit intermediate): a uniform interface at m!=0
  reflects each radial mode with the planar Fresnel coefficient OF ITS OWN
  POLARIZATION at that mode's local oblique angle
  theta = arcsin(gamma/(sqrt(eps) k0)) -- the cylindrical->planar
  correspondence, validated against the closed-form Fresnel (independent of
  both solvers).  Each checked mode is polarization-CLASSIFIED first (audit
  P3-64; the earlier min-over-both-coefficients form was necessary-not-
  sufficient -- a systematic TE/TM swap would have passed).  Scope caveat: on
  this nodal PEC basis the TM-like modes carry O(1) relative divergence and
  are dropped by the reldiv filter, so GATE 4a anchors the TE coefficient in
  practice; the TM anchor (and full multi-order discrimination) lives in the
  slower ``test_gate4.py`` grating comparison on the staggered basis.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.bor.bor_solve import _physical_propagating, build_layer, solve
from lumenairy.elements.bor.coupled_radial_eigensolver import _fd_grid, _pec_wall_ops
from lumenairy.elements.bor.zcascade import interface_smatrix

pytestmark = pytest.mark.slow      # eig-heavy BOR-PMM convergence tests


def _uni(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _ring(period, e_lo, e_hi, duty=0.5):
    def f(r):
        e = np.full_like(r, e_lo, dtype=complex)
        e[(r % period) < duty * period] = e_hi
        return e
    return f


def test_structured_stack_energy_staggered_default():
    """The staggered default: a ring-grating stack conserves R+T to machine
    precision (the same div-conforming basis production ``BORStack`` uses)."""
    m, R, N, k0 = 1, 4.0, 200, 2.0
    layers = [build_layer(m, R, N, _uni(2.0), k0),
              build_layer(m, R, N, _ring(0.8, 2.0, 6.0), k0, thickness=0.5),
              build_layer(m, R, N, _uni(2.0), k0)]
    res = solve(layers, k0)
    assert len(res["inc"]) >= 4                       # multiple physical channels
    assert np.max(np.abs(res["energy"] - 1.0)) < 1e-9


def test_structured_stack_energy_floor_nodal():
    """The legacy nodal basis conserves R+T only to the FD spurious-mode floor
    (~1-4%) -- and ONLY at small cells (it blows up beyond Rbig ~ several
    lambda; the staggered default is the fix)."""
    m, R, N, k0 = 1, 4.0, 200, 2.0
    layers = [build_layer(m, R, N, _uni(2.0), k0, basis="nodal"),
              build_layer(m, R, N, _ring(0.8, 2.0, 6.0), k0, thickness=0.5,
                          basis="nodal"),
              build_layer(m, R, N, _uni(2.0), k0, basis="nodal")]
    res = solve(layers, k0)
    assert len(res["inc"]) >= 4                       # multiple physical channels
    assert np.max(np.abs(res["energy"] - 1.0)) < 0.05  # within the ~4% floor


def _pol_fraction_nodal(m, R, N, k0, eps_val, L):
    """P_te[j] = <|E_phi|^2> / <|E_r|^2+|E_phi|^2+|E_z|^2> per mode column on
    the nodal PEC-wall basis (the ``build_layer`` basis) -- the classifier
    pattern of ``test_gate4._pol_fraction``, re-derived for the non-staggered
    grid.  TE-like (azimuthal E) -> 1; TM-like (E_r, E_z in the plane of
    incidence) -> 0.  Rebuilds the layer's E_z-elimination operators exactly
    as ``layer_modes(wall='pec')`` assembles them."""
    r, D, h = _fd_grid(R, N)
    D, Lap = _pec_wall_ops(D, h, N)
    ir = np.diag(1.0 / r)
    mr = m * ir
    A = D + ir
    Lm = Lap + ir @ D - (m ** 2) * np.diag(1.0 / r ** 2)
    Lei = np.linalg.inv(Lm + k0 ** 2 * np.diag(
        np.full(N, eps_val, dtype=complex)))
    wq = np.real(L["wq"])
    Pte = np.zeros(L["W"].shape[1])
    for j in range(L["W"].shape[1]):
        Er, Ephi = L["W"][:N, j], L["W"][N:, j]
        Ez = L["q"][j] * (Lei @ (1j * A @ Er - mr @ Ephi))
        e_r = np.sum(np.abs(Er) ** 2 * wq)
        e_p = np.sum(np.abs(Ephi) ** 2 * wq)
        e_z = np.sum(np.abs(Ez) ** 2 * wq)
        Pte[j] = e_p / max(e_r + e_p + e_z, 1e-300)
    return Pte


def test_gate4a_planar_fresnel_correspondence():
    """GATE 4a: m!=0 uniform-interface per-mode |S11| == the planar Fresnel
    coefficient of the mode's OWN polarization at its local oblique angle, to
    ~1e-3 (measured ~1e-5).  Audit P3-64: modes are pol-classified first --
    the old ``min(|s-rTM|, |s-rTE|)`` passed under a systematic TE/TM swap
    (|rTE-rTM| is 0.003..0.23 here, up to 230x the tolerance)."""
    m, R, N, k0 = 1, 5.0, 300, 2.0
    e1, e2 = 4.0, 2.25
    # the NODAL basis explicitly: _pol_fraction_nodal rebuilds that basis's
    # E_z-elimination operators (single cell-centered grid), and this gate's
    # documented TE-anchor scope is a nodal-basis property.
    La = build_layer(m, R, N, _uni(e1), k0, basis="nodal")
    Lb = build_layer(m, R, N, _uni(e2), k0, basis="nodal")
    S11 = interface_smatrix(La["W"], La["V"], Lb["W"], Lb["V"])[0]
    qa = La["q"]
    Pte = _pol_fraction_nodal(m, R, N, k0, e1, La)
    n_checked = 0
    for j in np.where(_physical_propagating(La, k0))[0]:
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
        # anchor against the classified polarization ONLY (test_gate4 cuts:
        # TE-like > 0.6, TM-like < 0.4; mixed modes have no planar anchor)
        if Pte[j] > 0.6:
            assert abs(s - rTE) < 1e-3, \
                f"TE-like mode {j} (P_te={Pte[j]:.2f}): |s-rTE|={abs(s-rTE):.2e}"
        elif Pte[j] < 0.4:
            assert abs(s - rTM) < 1e-3, \
                f"TM-like mode {j} (P_te={Pte[j]:.2f}): |s-rTM|={abs(s-rTM):.2e}"
        else:
            continue
        n_checked += 1
    assert n_checked >= 5


def test_gate4a_oblique_angles_span():
    """The m=1 modes sample a spread of local oblique angles (not all normal),
    so GATE 4a genuinely exercises the cylindrical-metric curvature."""
    m, R, N, k0 = 1, 5.0, 300, 2.0
    e1 = 4.0
    La = build_layer(m, R, N, _uni(e1), k0, basis="nodal")
    qa = La["q"]
    angles = []
    for j in np.where(_physical_propagating(La, k0))[0]:
        g2 = e1 * k0 ** 2 - qa[j].real ** 2
        if g2 > 0:
            angles.append(np.degrees(np.arcsin(np.sqrt(g2) / (np.sqrt(e1) * k0))))
    assert max(angles) - min(angles) > 20.0           # a real oblique spread
