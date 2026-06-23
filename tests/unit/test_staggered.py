"""Validate the Yee-staggered (div-conforming) discretization -- the cure for
the nodal solver's spurious-mode floor.

The nodal FD operator emits a ~91%-spurious modal sea (generic curl-curl
gradient modes, present even in a homogeneous layer); their real-q members leak
~1.5% (max 3.8%) of the cascade energy into unphysical channels, a floor that
does NOT decrease with N.  The Yee staggering (E_r on faces where D_r=eps E_r is
single-valued, E_phi/E_z on nodes) makes the discrete de Rham identity
``curl.grad == 0`` hold to machine precision, so the gradient null-space
collapses to a benign electrostatic branch out of the propagating window --
NO filtering, NO penalty, eig still 2N (complete square basis).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.bor.coupled_radial_eigensolver import (
    _fd_grid_staggered,
    radial_coupled_modes,
)
from lumenairy.elements.bor.fiber_oracle import fiber_modes
from lumenairy.elements.bor.zcascade import (
    interface_smatrix,
    layer_modes,
    propagation_smatrix,
    redheffer_star,
)

pytestmark = pytest.mark.slow      # eig-heavy BOR-PMM convergence tests


def test_derham_curl_grad_null():
    """curl_z(grad phi) == 0 on the staggered grid (the cure's mechanism): if
    this fails the face/node stencils are misaligned by half a cell."""
    N, Rbig, m = 200, 3.0, 1
    r_n, r_f, h, Dn2f, Df2n, An2f, Af2n = _fd_grid_staggered(Rbig, N)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        phi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        Er = Dn2f @ phi                                  # faces
        Ephi = (1j * m / r_n) * phi                      # nodes
        curlz = (1.0 / r_f) * (Dn2f @ (r_n * Ephi)) - (1j * m / r_f) * Er
        worst = max(worst, np.max(np.abs(curlz[1:-1])))
    assert worst < 1e-10


def test_spurious_sea_collapses():
    """Staggered V=4 fiber: ZERO spurious modes (vs ~365/400 nodal), and the
    known nodal spurious q~3.76 is GONE from the window, while the guided mode
    still matches the fiber oracle."""
    m, a, e1, e2, k0 = 1, 1.0, 6.0, 2.0, 2.0
    N, Rbig = 200, 8.0
    def eps(r):
        return np.where(r <= a, e1, e2).astype(complex)
    md = radial_coupled_modes(m, Rbig, N, eps, k0, staggered=True)
    rd = np.array([x["reldiv"] for x in md])
    q = np.array([x["q"] for x in md])
    assert np.sum(rd > 1.0) == 0                          # NO spurious at all
    lo, hi = np.sqrt(e2) * k0, np.sqrt(e1) * k0
    inwin = (np.abs(q.imag) < 1e-3) & (q.real > lo + 0.05) & (q.real < hi - 0.05)
    assert np.sum(inwin & (rd > 0.5)) == 0               # window is spurious-free
    qstrong = fiber_modes(m, a, e1, e2, k0)
    qstrong = qstrong[qstrong > 3.5][0]
    gq = q.real[inwin & (rd < 0.5)]
    assert np.min(np.abs(gq - qstrong)) < 3e-2           # guided mode preserved


def _ring(period, lo, hi):
    return lambda r: np.where((r % period) < 0.5 * period, hi, lo).astype(complex)


def _uni(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _cascade_energy(N):
    m, Rbig, k0 = 1, 4.0, 2.0
    Ls = layer_modes(m, Rbig, N, _uni(2.0), k0, staggered=True)
    Lr = layer_modes(m, Rbig, N, _ring(0.8, 2.0, 6.0), k0, staggered=True)
    Lt = layer_modes(m, Rbig, N, _uni(2.0), k0, staggered=True)
    S = redheffer_star(
        redheffer_star(interface_smatrix(Ls["W"], Ls["V"], Lr["W"], Lr["V"]),
                       propagation_smatrix(Lr["q"], 0.5)),
        interface_smatrix(Lr["W"], Lr["V"], Lt["W"], Lt["V"]))
    S11, S21 = S[0], S[2]
    q = Ls["q"]
    pin = np.where((np.abs(q.imag) < 1e-4) & (q.real > 0.1)
                   & (q.real < np.sqrt(2) * k0))[0]
    return max(abs(sum(abs(S11[jp, j]) ** 2 for jp in pin)
                   + sum(abs(S21[jp, j]) ** 2 for jp in pin) - 1) for j in pin)


def test_structured_cascade_energy_machine_precision():
    """The structured ring-grating cascade conserves energy to MACHINE PRECISION
    on the staggered grid (nodal floor was 3.8e-2, N-invariant)."""
    assert _cascade_energy(150) < 1e-9


def test_structured_energy_N_stable():
    """The cured energy stays at machine precision across N (it is no longer a
    floor -- the spurious leak is gone, not just refined away)."""
    assert _cascade_energy(150) < 1e-9
    assert _cascade_energy(300) < 1e-9
