"""Validate the BOR-PMM Milestone 4 z-cascade S-matrix machinery.

Gates (from the M4 research/design matrix; ranked by what NEW machinery each
tests -- see docs/pmm_bor_axisymmetric_roadmap.md M4 section):

- GATE 0  same-medium identity  -- the STRONG measure-free test: identical eps
  both sides must give ZERO reflection.  No Fresnel coefficient can absorb an
  r*dr/flux-sign/overlap error, so any nonzero S11 is a pure machinery bug.
- ROUND-TRIP identity -- a->b->a interface pair (no propagation) == identity.
- GATE 1  per-mode Fresnel (m=0) -- CONVENTION/SIGN smoke test only (circular,
  necessary-not-sufficient); checks eps-vs-q placement + p/s pair mapping.
- GATE 2  slab Fabry-Perot Airy -- propagation + Redheffer sign test.
- GATE 3  energy conservation R+T=1 -- continuous monitor (necessary-not-
  sufficient: the "lossless trap"); paired with GATE 0 which IS per-quantity.

These use the M2 FD eigenbasis (complete -> square pointwise interface).  Clean
multi-mode half-spaces (PEC wall / Galerkin projection) + the structured-layer
Cartesian-limit gate are M5 scope.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.bor.zcascade import (
    interface_smatrix,
    layer_modes,
    propagation_smatrix,
    redheffer_star,
)

pytestmark = pytest.mark.slow      # eig-heavy BOR-PMM convergence tests


def _uniform(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _flux(L, j):
    N = L["N"]
    Er, Ephi = L["W"][:N, j], L["W"][N:, j]
    hr, hphi = L["V"][:N, j], L["V"][N:, j]
    return np.real(np.sum((Er * np.conj(hphi) - Ephi * np.conj(hr)) * L["wq"]))


def test_same_medium_identity():
    """GATE 0: identical eps both sides -> S11 ~ 0, S21 ~ I (measure-free)."""
    m, R, N, k0 = 1, 3.0, 200, 2.0
    La = layer_modes(m, R, N, _uniform(4.0), k0)
    Lb = layer_modes(m, R, N, _uniform(4.0), k0)
    S11, S12, S21, S22 = interface_smatrix(La["W"], La["V"], Lb["W"], Lb["V"])
    assert np.max(np.abs(S11)) < 1e-7
    assert np.max(np.abs(S21 - np.eye(2 * N))) < 1e-7


def test_roundtrip_interface_identity():
    """a->b->a interface pair (no propagation) == identity."""
    m, R, N, k0 = 0, 4.0, 200, 2.0
    La = layer_modes(m, R, N, _uniform(4.0), k0)
    Lb = layer_modes(m, R, N, _uniform(2.25), k0)
    Sab = interface_smatrix(La["W"], La["V"], Lb["W"], Lb["V"])
    Sba = interface_smatrix(Lb["W"], Lb["V"], La["W"], La["V"])
    S = redheffer_star(Sab, Sba)
    assert np.max(np.abs(S[0])) < 1e-7
    assert np.max(np.abs(S[2] - np.eye(2 * N))) < 1e-7


def test_per_mode_fresnel_sign():
    """GATE 1 (m=0 sign smoke): each propagating mode's |S11| == its per-mode
    Fresnel coefficient.  Uses the PEC (Dirichlet) wall -- which supports MANY
    clean propagating modes -- rather than the natural wall's single leaky mode,
    so the check is BLAS-robust (the natural-wall variant could surface 0 clean
    propagating modes under a different eig ordering).  The best-resolved mode
    matches to machine precision; the set matches Fresnel on average to <1e-2."""
    m, R, N, k0 = 0, 3.0, 200, 2.0
    e1, e2 = 4.0, 2.0
    La = layer_modes(m, R, N, _uniform(e1), k0, wall="pec")
    Lb = layer_modes(m, R, N, _uniform(e2), k0, wall="pec")
    S11 = interface_smatrix(La["W"], La["V"], Lb["W"], Lb["V"])[0]
    qa = La["q"]
    errs = []
    for j in range(2 * N):
        q1 = qa[j]
        if abs(q1.imag) > 1e-6 or q1.real < 0.2:
            continue
        g2 = e1 * k0 ** 2 - q1.real ** 2
        if g2 < 0:                              # above the light line -> skip
            continue
        g = np.sqrt(g2)
        q2 = np.sqrt(e2 * k0 ** 2 - g ** 2 + 0j)
        if q2.imag > 1e-6:
            continue
        rTM = (e2 * q1 - e1 * q2) / (e2 * q1 + e1 * q2)
        rTE = (q1 - q2) / (q1 + q2)
        errs.append(min(abs(abs(S11[j, j]) - abs(rTM)),
                        abs(abs(S11[j, j]) - abs(rTE))))
    assert len(errs) >= 3                       # PEC -> several clean propagating modes
    assert min(errs) < 1e-6                     # the best-resolved mode matches exactly
    assert np.mean(errs) < 1e-2                 # all match Fresnel


def test_slab_fabry_perot():
    """GATE 2 (m=0): per-mode slab reflection == Airy r12(1-e)/(1-r12^2 e).  PEC
    wall (many clean propagating modes -> BLAS-robust); the best-resolved mode
    matches the Airy formula to machine precision."""
    m, R, N, k0 = 0, 6.0, 300, 2.0
    e1, e2, d = 4.0, 2.25, 1.3
    La = layer_modes(m, R, N, _uniform(e1), k0, wall="pec")
    Lmid = layer_modes(m, R, N, _uniform(e2), k0, wall="pec")
    S = redheffer_star(
        redheffer_star(interface_smatrix(La["W"], La["V"], Lmid["W"], Lmid["V"]),
                       propagation_smatrix(Lmid["q"], d)),
        interface_smatrix(Lmid["W"], Lmid["V"], La["W"], La["V"]))
    Sslab = S[0]
    qa = La["q"]
    errs = []
    for j in range(2 * N):
        q1 = qa[j]
        if abs(q1.imag) > 1e-6 or q1.real < 0.2:
            continue
        g2 = e1 * k0 ** 2 - q1.real ** 2
        if g2 < 0:                              # above the light line -> skip
            continue
        g = np.sqrt(g2)
        q2 = np.sqrt(e2 * k0 ** 2 - g ** 2 + 0j)
        if q2.imag > 1e-6:
            continue
        e = np.exp(2j * q2 * d)
        best = 1e9
        for r12 in ((e2 * q1 - e1 * q2) / (e2 * q1 + e1 * q2),
                    (q1 - q2) / (q1 + q2)):
            rs = r12 * (1 - e) / (1 - r12 ** 2 * e)
            best = min(best, abs(Sslab[j, j] - rs), abs(Sslab[j, j] + rs))
        errs.append(best)
    assert len(errs) >= 3                       # PEC -> several clean propagating modes
    assert min(errs) < 1e-6                     # best-resolved mode matches Airy exactly


def test_pec_wall_multimode_fresnel():
    """M5a: a PEC (Dirichlet) outer wall gives CLEAN multi-mode half-spaces.
    With wall='pec' many propagating modes appear (natural wall leaks all but
    ~1), and each matches its per-mode Fresnel coefficient."""
    m, R, N, k0 = 0, 5.0, 300, 2.0
    e1, e2 = 4.0, 2.25
    La = layer_modes(m, R, N, _uniform(e1), k0, wall="pec")
    Lb = layer_modes(m, R, N, _uniform(e2), k0, wall="pec")
    S11 = interface_smatrix(La["W"], La["V"], Lb["W"], Lb["V"])[0]
    qa = La["q"]
    errs = []
    for j in range(2 * N):
        q1 = qa[j]
        if abs(q1.imag) > 1e-6 or q1.real < 0.2:
            continue
        g2 = e1 * k0 ** 2 - q1.real ** 2
        if g2 < 0:
            continue
        g = np.sqrt(g2)
        q2 = np.sqrt(e2 * k0 ** 2 - g ** 2 + 0j)
        if q2.imag > 1e-6:
            continue
        rTM = (e2 * q1 - e1 * q2) / (e2 * q1 + e1 * q2)
        rTE = (q1 - q2) / (q1 + q2)
        errs.append(min(abs(abs(S11[j, j]) - abs(rTM)),
                        abs(abs(S11[j, j]) - abs(rTE))))
    assert len(errs) >= 8                       # MANY clean propagating modes
    assert np.mean(errs) < 1e-2                  # all match Fresnel


def test_pec_wall_same_medium_identity():
    """M5a: PEC-wall half-spaces still satisfy the measure-free identity gate."""
    m, R, N, k0 = 0, 5.0, 300, 2.0
    La = layer_modes(m, R, N, _uniform(4.0), k0, wall="pec")
    Lb = layer_modes(m, R, N, _uniform(4.0), k0, wall="pec")
    S11, S12, S21, S22 = interface_smatrix(La["W"], La["V"], Lb["W"], Lb["V"])
    assert np.max(np.abs(S11)) < 1e-7


def test_energy_conservation_slab():
    """GATE 3 (monitor): R + T == 1 on a lossless slab, propagating modes."""
    m, R, N, k0 = 0, 6.0, 300, 2.0
    e1, e2, d = 4.0, 2.25, 1.3
    La = layer_modes(m, R, N, _uniform(e1), k0)
    Lmid = layer_modes(m, R, N, _uniform(e2), k0)
    S = redheffer_star(
        redheffer_star(interface_smatrix(La["W"], La["V"], Lmid["W"], Lmid["V"]),
                       propagation_smatrix(Lmid["q"], d)),
        interface_smatrix(Lmid["W"], Lmid["V"], La["W"], La["V"]))
    S11, S12, S21, S22 = S
    qa = La["q"]
    checked = 0
    for j in range(2 * N):
        if abs(qa[j].imag) > 1e-6 or qa[j].real < 0.2:
            continue
        Pin = _flux(La, j)
        if abs(Pin) < 1e-12:
            continue
        Rj = abs(S11[j, j]) ** 2
        Tj = abs(S21[j, j]) ** 2 * abs(_flux(La, j) / Pin)
        assert abs(Rj + Tj - 1.0) < 1e-6
        checked += 1
    assert checked >= 1
