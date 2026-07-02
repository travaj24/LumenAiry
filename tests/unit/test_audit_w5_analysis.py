"""Audit v5.17.0 Wave 5 -- analysis cluster (zernike.py + ao.py).

P2-02 (physics, BEHAVIOR CHANGE)
    ``normalization='Noll'`` used to multiply every m < 0 (sine)
    coefficient by -1 on the claim that "Noll's sine convention is
    opposite to OSA".  That claim is false: Noll 1976 Eq. (2) defines
    the sine modes as ``sqrt(2(n+1)) R_n^m(rho) sin(m theta)`` with
    m > 0 (positive sine), and OSA/ANSI Z80.28 defines ``Z_n^m`` for
    m < 0 as ``-N R sin(m theta) = +N R sin(|m| theta)`` -- the SAME
    positive sine.  The conventions differ only in single-index
    ordering.  The old factor therefore returned the NEGATIVE of the
    true Noll coefficient for every sine mode, matching no published
    convention.  The pins below are NON-CIRCULAR: the input OPD maps
    are built from hand-written Noll 1976 Table I polynomials (literal
    numpy expressions, no library calls).

P2-01 (memory/perf)
    ``DeformableMirror.fit_phase``'s "streamed" large-DM branch
    materialised ALL influence-function columns (full design-matrix
    memory, defeating its own docstring contract) and formed the
    normal matrix via n2*(n2+1)/2 Python-level np.dot calls.  Fixed to
    genuinely stream: normal equations accumulated over horizontal
    grid-row bands with BLAS gemms (bounded scratch), or a single gemm
    on views of the cached IF basis.  Pins: streamed result matches
    the dense-lstsq path on the FITTED PHASE (conditioning caveat) and
    peak traced memory stays far below the full design matrix.

Author: audit wave 5, 2026-07-02.
"""
from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from lumenairy.analysis import ao as ao_mod
from lumenairy.analysis.ao import DeformableMirror
from lumenairy.analysis.zernike import zernike_decompose

# ======================================================================
# P2-02 -- Noll normalization signs
# ======================================================================

@pytest.fixture(scope='module')
def pupil_grid():
    N = 96
    dx = 5e-6
    aperture = 0.6 * N * dx
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r = 0.5 * aperture
    rho = np.hypot(X, Y) / r
    theta = np.arctan2(Y, X)
    return {'dx': dx, 'aperture': aperture, 'rho': rho, 'theta': theta}


# Hand-written Noll 1976 Table I polynomials (J. Opt. Soc. Am. 66,
# 207-211).  Written LITERALLY -- no zernike_polynomial calls -- so the
# oracle is independent of the library's internal basis.  Each maps to
# the OSA single index j_OSA = (n*(n+2) + m) / 2 with m < 0 for the
# sine form.
def _noll_Z3_tilt_y(rho, th):        # Noll j=3: sqrt(4) rho sin(theta)
    return 2.0 * rho * np.sin(th)


def _noll_Z5_oblique_astig(rho, th):  # Noll j=5: sqrt(6) rho^2 sin(2 theta)
    return np.sqrt(6.0) * rho ** 2 * np.sin(2.0 * th)


def _noll_Z7_vertical_coma(rho, th):  # Noll j=7: sqrt(8)(3 rho^3 - 2 rho) sin(theta)
    return np.sqrt(8.0) * (3.0 * rho ** 3 - 2.0 * rho) * np.sin(th)


def _noll_Z6_vertical_astig(rho, th):  # Noll j=6: sqrt(6) rho^2 cos(2 theta)
    return np.sqrt(6.0) * rho ** 2 * np.cos(2.0 * th)


def test_noll_sine_coefficients_match_noll_1976_table(pupil_grid):
    """Decomposing amp * (hand-written Noll sine polynomial) with
    normalization='Noll' must recover +amp (NOT -amp, the pre-fix
    output) at the corresponding OSA index.

    OSA index bookkeeping: (n, m) = (1, -1) -> j=1; (2, -2) -> j=3;
    (3, -1) -> j=7 via j = (n(n+2) + m)/2.
    """
    rho = pupil_grid['rho']
    th = pupil_grid['theta']
    cases = [
        ('Noll Z3 tilt-Y', _noll_Z3_tilt_y, 1, 1.3e-7),
        ('Noll Z5 oblique astig', _noll_Z5_oblique_astig, 3, -4.0e-8),
        ('Noll Z7 vertical coma', _noll_Z7_vertical_coma, 7, 2.5e-8),
    ]
    for label, poly, j_osa, amp in cases:
        opd = np.where(rho <= 1.0, amp * poly(rho, th), 0.0)
        c, _ = zernike_decompose(
            opd, pupil_grid['dx'], pupil_grid['aperture'],
            n_modes=15, normalization='Noll')
        assert c[j_osa] == pytest.approx(amp, rel=1e-6), (
            f'{label}: expected Noll coefficient {amp:+.3e} at OSA '
            f'j={j_osa}, got {c[j_osa]:+.3e} (a sign flip means the '
            f'fabricated pre-fix convention is back)')
        # All other fitted modes stay negligible.
        rest = np.delete(c, j_osa)
        assert np.max(np.abs(rest)) < 1e-6 * abs(amp) + 1e-18


def test_noll_mixed_sine_cosine_matches_hand_oracle(pupil_grid):
    """A mixed sine + cosine OPD built purely from hand-written Noll
    Table I polynomials must come back with BOTH coefficients upright
    under normalization='Noll' (and identical under 'OSA', since the
    polynomials coincide)."""
    rho = pupil_grid['rho']
    th = pupil_grid['theta']
    a_sin, a_cos = 6.0e-8, -9.0e-8
    opd = np.where(
        rho <= 1.0,
        a_sin * _noll_Z5_oblique_astig(rho, th)
        + a_cos * _noll_Z6_vertical_astig(rho, th),
        0.0)
    for norm in ('Noll', 'OSA'):
        c, _ = zernike_decompose(
            opd, pupil_grid['dx'], pupil_grid['aperture'],
            n_modes=10, normalization=norm)
        # OSA j: (2,-2) -> 3 (sine), (2,+2) -> 5 (cosine).
        assert c[3] == pytest.approx(a_sin, rel=1e-6), norm
        assert c[5] == pytest.approx(a_cos, rel=1e-6), norm


# ======================================================================
# P2-01 -- fit_phase streamed branch
# ======================================================================

def _make_target(dm: DeformableMirror, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cmd = rng.standard_normal((dm.n_actuators, dm.n_actuators))
    dm.set_command(cmd)
    target = dm.phase()
    dm.set_command(np.zeros_like(cmd))
    return target


def _fitted_phase(dm: DeformableMirror, coeffs: np.ndarray) -> np.ndarray:
    dm.set_command(coeffs)
    phi = dm.phase()
    dm.set_command(np.zeros((dm.n_actuators, dm.n_actuators)))
    return phi


@pytest.mark.parametrize('cache_basis', [False, True],
                         ids=['band-streamed', 'cached-gemm'])
def test_streamed_fit_phase_matches_dense_path(monkeypatch, cache_basis):
    """Forcing the large-DM branch (ceiling monkeypatched down) must
    reproduce the dense-lstsq fit.  Compared on the FITTED PHASE, not
    raw coefficients, per the least-squares conditioning caveat.
    Covers both streamed sub-paths: uncached row-band accumulation and
    cached-basis single-gemm."""
    n_act, N, dx = 6, 64, 1e-3
    dm = DeformableMirror(n_actuators=n_act, pitch=N * dx / n_act,
                          dx=dx, N=N, cache_basis=cache_basis)
    target = _make_target(dm)

    # Dense reference path (real ceiling: 64^2*36*8 = 1.2 MB << 128 MB).
    coeffs_dense = dm.fit_phase(target)
    phase_dense = _fitted_phase(dm, coeffs_dense)

    # Force the streamed branch: bytes_design (1.2 MB) > ceiling // 4.
    monkeypatch.setattr(ao_mod, '_DEFAULT_CACHE_CEILING_BYTES',
                        1 * 1024 * 1024)
    coeffs_str = dm.fit_phase(target)
    phase_str = _fitted_phase(dm, coeffs_str)

    scale = float(np.max(np.abs(target)))
    err = float(np.max(np.abs(phase_str - phase_dense))) / scale
    assert err < 1e-10, (
        f'streamed fitted phase deviates from dense path by {err:.3e} '
        f'(relative to max |target|)')
    # Well-posed geometry here, so coefficients should agree too.
    np.testing.assert_allclose(coeffs_str, coeffs_dense,
                               rtol=1e-8, atol=1e-12)


def test_streamed_fit_phase_memory_actually_bounded(monkeypatch):
    """The pre-fix branch materialised every IF column: for this
    uncached 10x10-actuator / 256-grid case that is the FULL 52.4 MB
    design matrix.  The fixed row-band accumulation must keep the
    traced peak far below that (band scratch is bounded by
    ceiling // 16 = 0.5 MB here, plus the 80 kB normal matrix and
    small temporaries)."""
    n_act, N, dx = 10, 256, 1e-3
    dm = DeformableMirror(n_actuators=n_act, pitch=N * dx / n_act,
                          dx=dx, N=N, cache_basis=False)
    target = _make_target(dm)
    bytes_design = (N ** 2) * (n_act ** 2) * 8  # 52.4 MB

    monkeypatch.setattr(ao_mod, '_DEFAULT_CACHE_CEILING_BYTES',
                        8 * 1024 * 1024)  # gate: 52.4 MB > 2 MB
    tracemalloc.start()
    coeffs = dm.fit_phase(target)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 0.25 * bytes_design, (
        f'fit_phase traced peak {peak/1e6:.1f} MB is not << full '
        f'design matrix {bytes_design/1e6:.1f} MB -- streamed branch '
        f'is materialising columns again')
    # And it still fits the target well (target lies in the IF span).
    phase_fit = _fitted_phase(dm, coeffs)
    scale = float(np.max(np.abs(target)))
    assert float(np.max(np.abs(phase_fit - target))) / scale < 1e-8


def test_streamed_fit_phase_rank_deficient_does_not_blow_up(monkeypatch):
    """Degenerate geometry (huge inter-actuator coupling -> nearly
    identical influence functions -> rank-deficient normal matrix)
    must yield a finite fit, not a LinAlgError or 1e60-scale garbage:
    the streamed branch solves the normal system with lstsq
    (minimum-norm), not np.linalg.solve."""
    n_act, N, dx = 5, 48, 1e-3
    dm = DeformableMirror(n_actuators=n_act, pitch=0.05 * N * dx / n_act,
                          dx=dx, N=N, inter_actuator_coupling=0.99,
                          cache_basis=False)
    target = _make_target(dm)
    monkeypatch.setattr(ao_mod, '_DEFAULT_CACHE_CEILING_BYTES', 4096)
    coeffs = dm.fit_phase(target)
    assert np.all(np.isfinite(coeffs))
    phase_fit = _fitted_phase(dm, coeffs)
    scale = float(np.max(np.abs(target))) or 1.0
    # The reconstructed phase must still approximate the target (it is
    # in the span of the influence functions by construction).
    assert float(np.max(np.abs(phase_fit - target))) / scale < 1e-6
