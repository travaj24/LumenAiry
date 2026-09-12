"""VERIFY-A5 -- independent re-pins for K9 (Rayleigh-Sommerfeld routing) and
K16 (Richards-Wolf ``E_z`` sign) on fixtures and polarisations WP-A5 did not
use.

Two gaps in WP-A5's own pins motivated these:

* ``TestK9RayleighSommerfeldNearField::test_far_field_is_bit_identical_to_the
  _historical_kernel`` compares ``kernel='auto'`` with ``kernel='spatial'``,
  which is what the routing code is defined to do -- it cannot detect a change
  in the spatial kernel itself, and it exercises no oracle.  The pins here use
  an INDEPENDENT reference (an 8x-zero-padded analytic RS-I transfer-function
  evaluation written out in this file, never a library call), on odd N, on an
  anamorphic pitch, and at complex64 -- none of which the WP's fixtures cover.
* ``TestK16RichardsWolfEzSign`` pins only the x-polarised focus.  ``E_z`` is
  the component whose SIGN was wrong; its y-polarised and circular
  counterparts are the other two states a caller reaches through the public
  ``polarization`` argument and nothing pinned them.

Author: VERIFY-A5.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.propagators.rs import (
    _rs_alias_free_distance,
    rayleigh_sommerfeld_propagate,
)
from lumenairy.propagators.vector_diffraction import richards_wolf_focus


def _grid(N, d):
    return (np.arange(N) - N / 2) * d


def _linear_convolution_reference(E, z, lam, dx, dy, pad=8):
    """Independent reference for the RS-I operator, written here rather than
    taken from the library: zero-pad the SAME samples to ``pad*N`` and
    multiply by the analytic transfer function
    ``exp(i k z sqrt(1 - (lam f)^2))`` with the evanescent set zeroed.

    At ``pad = 8`` the periodised impulse response is negligible over the
    valid region, so this is the linear convolution the propagator claims to
    compute, free of both the 2N wrap-around and of any point-sampled kernel.
    Its own floor is measured by comparing ``pad = 8`` against ``pad = 16``:
    3.4e-15 / 9.8e-11 / 8.5e-12 / 5.8e-15 on the four fixtures below (the
    two larger values are the coarse-sampled Gaussian's own tail, not the
    reference's arithmetic).
    """
    Ny, Nx = E.shape
    My, Mx = pad * Ny, pad * Nx
    y0, x0 = (My - Ny) // 2, (Mx - Nx) // 2
    Ep = np.zeros((My, Mx), dtype=np.complex128)
    Ep[y0:y0 + Ny, x0:x0 + Nx] = E
    FX, FY = np.meshgrid(np.fft.fftfreq(Mx, dx), np.fft.fftfreq(My, dy),
                         indexing='xy')
    arg = 1.0 - (lam * FX) ** 2 - (lam * FY) ** 2
    H = np.where(arg > 0,
                 np.exp(2j * np.pi * z / lam * np.sqrt(np.maximum(arg, 0.0))),
                 0.0)
    return np.fft.ifft2(np.fft.fft2(Ep) * H)[y0:y0 + Ny, x0:x0 + Nx]


def _gauss(Ny, Nx, dy, dx, w0, dtype=np.complex128):
    X, Y = np.meshgrid(_grid(Nx, dx), _grid(Ny, dy), indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(dtype)


# (Ny, Nx, dy, dx, lam, w0, z, pre_fix_relL2, pre_fix_power)
K9_NEAR = [
    (65, 65, 1.5e-6, 1.5e-6, 532e-9, 5e-6, 30e-6, 4.554, 21.394),    # odd N
    (64, 64, 2.0e-6, 1.0e-6, 532e-9, 9e-6, 40e-6, 3.851, 15.173),    # dy != dx
    (48, 48, 1.0e-6, 1.0e-6, 633e-9, 6e-6, 20e-6, 2.028, 5.273),
    (127, 127, 0.8e-6, 0.8e-6, 488e-9, 5e-6, 60e-6, 1.925, 4.705),   # odd N
]


class TestK9AgainstAnIndependentLinearConvolution:

    @pytest.mark.parametrize('Ny,Nx,dy,dx,lam,w0,z,pre_rel,pre_pow', K9_NEAR)
    def test_near_field_default_matches_the_reference(
            self, Ny, Nx, dy, dx, lam, w0, z, pre_rel, pre_pow):
        """Bar: relative L2 < 1e-5 against the 8x-pad reference above.

        Derivation.  The reference's own floor is 3.4e-15 to 9.8e-11 on
        these four fixtures (pad 8 vs pad 16, quoted per fixture in
        ``_linear_convolution_reference``).  The default's measured
        distance from it is 2.3e-14 / 2.7e-9 / 2.2e-10 / 4.2e-14 --
        i.e. at the reference's own floor on every one.  The bar 1e-5 sits
        3.6 decades above the worst measured value and 5.3 decades below
        the PRE-FIX distances, re-measured here against the pre-v5.46
        module itself: 4.554 / 3.851 / 2.028 / 1.925.  Both gaps are
        stated, per TESTING_STANDARDS restatement 5.
        """
        E = _gauss(Ny, Nx, dy, dx, w0)
        assert z < max(_rs_alias_free_distance(Nx, dx, lam),
                       _rs_alias_free_distance(Ny, dy, lam)), (
            'fixture must sit in the aliasing regime the finding is about')
        ref = _linear_convolution_reference(E, z, lam, dx, dy)
        got = np.asarray(rayleigh_sommerfeld_propagate(E, z, lam, dx, dy=dy))
        rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        assert rel < 1e-5, (
            f'relative L2 {rel:.3e} against an 8x-pad analytic reference '
            f'at ({Ny},{Nx}) dy={dy:.2e} dx={dx:.2e} z={z:.2e} '
            f'(pre-fix {pre_rel}).')

    @pytest.mark.parametrize('Ny,Nx,dy,dx,lam,w0,z,pre_rel,pre_pow', K9_NEAR)
    def test_near_field_default_conserves_energy(
            self, Ny, Nx, dy, dx, lam, w0, z, pre_rel, pre_pow):
        """Bar: |P_out/P_in - 1| < 1e-4.

        Derivation.  The transfer kernel is unitary on the propagating set
        and these Gaussians put nothing at the grid Nyquist
        (``exp(-(pi w0 f_N)^2) <= 1e-24`` on all four), so the only loss is
        FFT round-off; measured 1.000000 to the six digits printed on every
        fixture.  1e-4 is ~4 decades above round-off and 4.7-5.3 decades
        below the pre-fix ratios 21.394 / 15.173 / 5.273 / 4.705, which are
        the energy the point-sampled kernel CREATED.
        """
        E = _gauss(Ny, Nx, dy, dx, w0)
        got = np.asarray(rayleigh_sommerfeld_propagate(E, z, lam, dx, dy=dy))
        ratio = float(np.sum(np.abs(got) ** 2) / np.sum(np.abs(E) ** 2))
        assert abs(ratio - 1.0) < 1e-4, (
            f'P_out/P_in = {ratio:.6f} (pre-fix {pre_pow}).')

    @pytest.mark.parametrize('Ny,Nx,dy,dx,lam,w0,z,pre_rel,pre_pow', K9_NEAR)
    def test_complex64_stays_at_its_own_floor(
            self, Ny, Nx, dy, dx, lam, w0, z, pre_rel, pre_pow):
        """The routed default must hold at single precision too -- the WP's
        K9 fixtures are all complex128.

        Bar: relative L2 < 1e-5 and |P_out/P_in - 1| < 1e-4.  Measured
        1.26e-7 / 1.17e-7 / 1.27e-7 / 3.78e-7, i.e. the complex64 epsilon
        (1.2e-7) as it should be, 1.4 decades below the bar; power
        1.000000 / 1.000000 / 1.000000 / 0.999999.
        """
        E = _gauss(Ny, Nx, dy, dx, w0, dtype=np.complex64)
        ref = _linear_convolution_reference(E.astype(np.complex128), z, lam,
                                            dx, dy)
        got = np.asarray(rayleigh_sommerfeld_propagate(E, z, lam, dx, dy=dy))
        assert got.dtype == np.complex64
        rel = float(np.linalg.norm(got.astype(np.complex128) - ref)
                    / np.linalg.norm(ref))
        ratio = float(np.sum(np.abs(got.astype(np.complex128)) ** 2)
                      / np.sum(np.abs(E.astype(np.complex128)) ** 2))
        assert rel < 1e-5, f'complex64 relative L2 {rel:.3e}'
        assert abs(ratio - 1.0) < 1e-4, f'complex64 P_out/P_in {ratio:.6f}'

    @pytest.mark.parametrize('N,dx,lam,w0', [
        (65, 1.0e-6, 633e-9, 6e-6), (96, 0.7e-6, 633e-9, 6e-6),
        (128, 0.5e-6, 532e-9, 4e-6),
    ])
    def test_the_two_branches_agree_at_the_switch(self, N, dx, lam, w0):
        """Continuity of the routed default: a caller sweeping ``z`` across
        ``z_crit`` must not see a step.

        Bar: relative L2 between the two branches at ``z = z_crit`` < 1e-8.
        Derivation: measured 1.16e-13 / 8.40e-14 / 8.05e-14 on these three
        grids, against each arm's OWN distance from the 8x-pad reference
        (8.6e-14 / 1.3e-13 / 1.7e-13) -- i.e. both branches and their
        difference sit together at the FFT round-off floor, so the step is
        not observable.  1e-8 sits ~5 decades above that floor and ~8
        decades below the smallest real discontinuity worth catching (the
        1.9e-2 wrap-around error the routing exists to avoid).
        """
        zc = _rs_alias_free_distance(N, dx, lam)
        E = _gauss(N, N, dx, dx, w0)
        t = np.asarray(rayleigh_sommerfeld_propagate(E, zc, lam, dx,
                                                      kernel='transfer'))
        s = np.asarray(rayleigh_sommerfeld_propagate(E, zc, lam, dx,
                                                      kernel='spatial'))
        step = float(np.linalg.norm(t - s) / np.linalg.norm(s))
        assert step < 1e-8, (
            f'the two RS discretisations differ by {step:.3e} at z_crit = '
            f'{zc:.3e} m (N={N}, dx={dx:.2e}); the default switches between '
            f'them there.')

    def test_auto_selects_the_documented_branch_either_side(self):
        """The routing rule itself, as a decision: strictly below the
        threshold the default is the transfer kernel, at and above it the
        spatial one.  ``kernel='spatial'`` raising below the threshold is
        what makes the lower half observable."""
        N, dx, lam, w0 = 64, 1e-6, 633e-9, 6e-6
        zc = _rs_alias_free_distance(N, dx, lam)
        E = _gauss(N, N, dx, dx, w0)
        below = np.asarray(rayleigh_sommerfeld_propagate(E, 0.5 * zc, lam, dx))
        below_t = np.asarray(rayleigh_sommerfeld_propagate(
            E, 0.5 * zc, lam, dx, kernel='transfer'))
        assert below.tobytes() == below_t.tobytes()
        above = np.asarray(rayleigh_sommerfeld_propagate(E, 2.0 * zc, lam, dx))
        above_s = np.asarray(rayleigh_sommerfeld_propagate(
            E, 2.0 * zc, lam, dx, kernel='spatial'))
        assert above.tobytes() == above_s.tobytes()


class TestK16EzSignForYAndCircularPolarisation:
    """K16 fixed the ``E_z`` sign; WP-A5 pinned it for ``polarization='x'``
    only.  ``E_z`` is odd in the polarisation azimuth, so the y and circular
    states are where a half-fix or a partial revert would show."""

    NA, F, LAM, NP, DXP = 0.7, 2e-3, 488e-9, 128, 25e-6

    def _pupil(self):
        Xp, Yp = np.meshgrid(_grid(self.NP, self.DXP),
                             _grid(self.NP, self.DXP), indexing='xy')
        return (np.hypot(Xp, Yp) <= self.F * self.NA).astype(complex)

    def _focus(self, pol):
        return richards_wolf_focus(self._pupil(), self.LAM, self.NA, self.F,
                                   self.DXP, polarization=pol)

    def test_y_polarised_focus_has_the_same_sign_rotated_by_90_degrees(self):
        """Novotny & Hecht eq. 3.66 with the polarisation along y:
        ``Im(E_z/E_y) < 0`` just off axis on the ``+y`` side, ``E_z`` odd in
        ``y`` and identically zero on the ``x`` axis.

        The sign is a DECISION, so the only bar is that the quantity is far
        from zero: measured ``Im(E_z/E_y) = -0.7801`` with a real part of
        -8.2e-17 (purely imaginary, as the closed form requires), so
        ``|Im| > 0.05`` has 1.2 decades of margin.  Pre-fix this read
        +0.7801.  The symmetry bars are FFT round-off: measured 1.19e-15
        (odd in y) and 2.43e-19 (zero on the x axis); bar 1e-12, ~3 decades
        above the round-off and ~12 below any real asymmetry.
        """
        Ex, Ey, Ez, _xf, _yf = self._focus('y')
        c = self.NP // 2
        r = complex(Ez[c + 1, c] / Ey[c + 1, c])
        assert abs(r.imag) > 0.05, f'fixture broken: |Im| = {abs(r.imag):.4f}'
        assert r.imag < 0.0, (
            f'Im(E_z/E_y) = {r.imag:+.6f} at y_f > 0 on a y-polarised focus; '
            f'Novotny-Hecht eq. 3.66 requires it NEGATIVE (pre-fix +0.7801).')
        assert complex(Ez[c - 1, c] / Ey[c - 1, c]).imag > 0.0
        odd = abs(Ez[c + 3, c] + Ez[c - 3, c]) / abs(Ez[c + 3, c])
        on_axis = abs(Ez[c, c + 3]) / np.abs(Ez).max()
        assert odd < 1e-12, f'E_z not odd in y: {odd:.3e}'
        assert on_axis < 1e-12, f'E_z not zero on the x axis: {on_axis:.3e}'
        # x-polarised and y-polarised focal fields are each other's 90-degree
        # rotation; on a symmetric pupil that is an exact transpose.
        _Ex2, _Ey2, Ez2, _a, _b = self._focus('x')
        mirror = float(np.max(np.abs(Ez - Ez2.T)) / np.abs(Ez2).max())
        assert mirror < 1e-12, (
            f'y-polarised E_z is not the transpose of the x-polarised one '
            f'({mirror:.3e}); the two differ by the sign convention this '
            f'finding is about.')

    def test_circular_polarisation_carries_the_expected_ez_vortex(self):
        """For circular input ``e_z`` picks up ``exp(i phi)``: ``E_z`` is a
        first-order vortex -- identically zero on axis, non-zero around it.
        Linearity gives the exact cross-check ``E_z(circ) = (E_z(x) +
        i E_z(y))/sqrt(2)``, which a sign flip on ONE state would break.

        Bars: on-axis ``|E_z|/max|E_z| < 1e-12`` (measured 3.03e-18, FFT
        round-off, ~6 decades of margin below the bar and ~12 below the
        0.5-ish value a non-vortex field would show); linearity residual
        < 1e-12 relative (measured 0.0 -- the same arithmetic path).
        """
        _Exc, _Eyc, Ezc, _a, _b = self._focus('circular')
        _Ex, _Ey, Ezx, _c, _d = self._focus('x')
        _Ex2, _Ey2, Ezy, _e, _f = self._focus('y')
        c = self.NP // 2
        peak = float(np.abs(Ezc).max())
        assert abs(Ezc[c, c]) / peak < 1e-12, (
            f'E_z should vanish on axis for a circularly polarised focus; '
            f'got {abs(Ezc[c, c]) / peak:.3e} of its own maximum.')
        assert (float(np.abs(Ezc[c, c + 3]))
                > 1e6 * float(np.abs(Ezc[c, c]))), (
            'fixture broken: E_z is flat, so the on-axis null means nothing. '
            'Measured ratio 8.9e15 (32.0 three pixels off axis against '
            '3.6e-15 on it); the bar sits 9 decades below that and 6 above '
            'a field with no null at all.')
        lin = (Ezx + 1j * Ezy) / np.sqrt(2.0)
        res = float(np.max(np.abs(Ezc - lin)) / peak)
        assert res < 1e-12, (
            f'E_z(circular) != (E_z(x) + i E_z(y))/sqrt(2): residual '
            f'{res:.3e}; one polarisation state carries a different sign.')
