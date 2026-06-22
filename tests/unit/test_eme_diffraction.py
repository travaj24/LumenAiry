"""Validate the scalar mode-matching diffraction MATH and record the structured
non-convergence finding (see ``eme_diffraction`` module docstring).

WHAT IS LOCKED HERE:
  * a UNIFORM layer reproduces the analytic scalar slab (machine precision at
    normal incidence, FD accuracy ~1e-4 at oblique), with energy R + T = 1
    exactly -- the mode-matching algebra is exact;
  * a STRUCTURED layer does NOT converge (efficiencies wander, energy strays from
    1) -- the documented dead end (real-space modes are not an order-space basis;
    use rcwa_efficiency_2d / pmm_efficiency_2d for efficiencies).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pytest

from lumenairy.elements.eme import diffraction_fd, strips_to_eps_xy

pytestmark = pytest.mark.slow      # eig-heavy EME convergence tests


def _analytic_slab(eps_sup, eps_l, eps_sub, depth, k0, kx0, ky0):
    """Scalar slab (continuity of psi and d psi/dz; admittance = kz) via Airy."""
    def kz(e):
        return np.sqrt(e * k0 ** 2 - kx0 ** 2 - ky0 ** 2 + 0j)
    k0s, k1, k2 = kz(eps_sup), kz(eps_l), kz(eps_sub)
    r01, r12 = (k0s - k1) / (k0s + k1), (k1 - k2) / (k1 + k2)
    t01, t12 = 2 * k0s / (k0s + k1), 2 * k1 / (k1 + k2)
    ph = np.exp(2j * k1 * depth)
    r = (r01 + r12 * ph) / (1 + r01 * r12 * ph)
    t = (t01 * t12 * np.exp(1j * k1 * depth)) / (1 + r01 * r12 * ph)
    return abs(r) ** 2, (k2.real / k0s.real) * abs(t) ** 2


def test_uniform_layer_matches_analytic_slab_normal():
    """Uniform layer, normal incidence -> exact scalar slab, energy = 1."""
    Lx = Ly = 1.0
    Nx = Ny = 24
    k0 = 4.0
    eps_sup, eps_l, eps_sub, depth = 1.0, 4.0, 2.25, 0.5
    R_an, T_an = _analytic_slab(eps_sup, eps_l, eps_sub, depth, k0, 0.0, 0.0)
    eps_xy = np.full((Nx, Ny), eps_l, dtype=complex)
    res = diffraction_fd(eps_xy, Lx, Ly, Nx, Ny, k0, eps_sup, eps_sub, depth,
                         2, 2, kx0=0.0, ky0=0.0)
    i00 = res["orders"].index((0, 0))
    assert abs(res["R"][i00] - R_an) < 1e-6
    assert abs(res["T"][i00] - T_an) < 1e-6
    assert abs(res["energy"] - 1.0) < 1e-9
    off = max(np.max(np.delete(res["R"], i00)), np.max(np.delete(res["T"], i00)))
    assert off < 1e-12                       # a uniform layer scatters no orders


def test_uniform_layer_matches_analytic_slab_oblique():
    """Uniform layer, oblique incidence -> exact scalar slab, energy = 1."""
    Lx = Ly = 2.0
    Nx = Ny = 24
    k0 = 5.0
    eps_sup, eps_l, eps_sub, depth = 1.0, 4.0, 2.25, 0.4
    kx0, ky0 = 0.0, 2.0
    R_an, T_an = _analytic_slab(eps_sup, eps_l, eps_sub, depth, k0, kx0, ky0)
    eps_xy = np.full((Nx, Ny), eps_l, dtype=complex)
    res = diffraction_fd(eps_xy, Lx, Ly, Nx, Ny, k0, eps_sup, eps_sub, depth,
                         2, 2, kx0=kx0, ky0=ky0)
    i00 = res["orders"].index((0, 0))
    # to FD accuracy: the (0,0)-coupled mode is exp(i ky0 y), whose FD eigenvalue
    # carries O(h^2) error (vs the exact flat mode at normal incidence)
    assert abs(res["R"][i00] - R_an) < 1e-3
    assert abs(res["T"][i00] - T_an) < 1e-3
    assert abs(res["energy"] - 1.0) < 1e-9     # energy still conserved exactly


def test_structured_layer_does_not_converge():
    """Records the finding: a structured layer's efficiencies do NOT converge and
    energy strays from 1 as the order/mode count grows (the documented dead end).
    A guard test -- if some future change makes this CONVERGE cleanly, revisit the
    module's negative-result conclusion."""
    Lx = Ly = 2.0
    Nx = 24
    k0 = 5.0
    eps_sup, eps_sub, depth = 1.0, 2.25, 0.4
    kx0, ky0 = 0.0, 2.0
    xg = (np.arange(Nx) + 0.5) / Nx * Lx
    block = np.where((xg >= 0.5) & (xg < 1.3), 6.0, 1.0)
    strips = [(np.full(Nx, 1.0), 0.7), (block, 0.6), (np.full(Nx, 1.0), 0.7)]
    eps_xy = strips_to_eps_xy(strips, Lx, Nx, Ly, 96)
    energies, t00 = [], []
    for M in (1, 2, 3, 4):
        res = diffraction_fd(eps_xy, Lx, Ly, Nx, 96, k0, eps_sup, eps_sub, depth,
                             M, M, kx0=kx0, ky0=ky0)
        energies.append(res["energy"])
        t00.append(res["T"][res["orders"].index((0, 0))])
    # NOT convergent: energy departs from 1 and T_00 keeps moving as M grows
    assert max(abs(e - 1.0) for e in energies) > 0.05      # energy strays
    assert max(t00) - min(t00) > 0.05                      # efficiency wanders
