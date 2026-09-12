"""WP-A5 regression pins -- AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.

Covers the propagator-kernel findings K1-K24 (report sections 3, 3.1,
3.2).  Every numeric bar below carries its oracle, that oracle's own
error floor, the measured pre-fix and post-fix values, and the decades
of gap on each side, per ``docs/TESTING_STANDARDS.md``.

Oracles used here are INDEPENDENT of the code under test:

* an exact Hankel angular-spectrum quadrature for a Gaussian (Gauss-
  Legendre over spatial frequency; no FFT, no grid) -- for the
  Rayleigh-Sommerfeld near field;
* a radial Bessel-integral Debye-Wolf form evaluated with
  ``scipy.integrate.quad`` -- for the Richards-Wolf ``E_z`` sign;
* ``scipy.special.jv`` -- for the backend Bessel dispatch;
* closed-form geometric / algebraic identities (norm preservation,
  transversality, energy inside a window) elsewhere.

Author: Andrew Traverso -- WP-A5.
"""
from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest
import scipy.special as _sp
from scipy.integrate import quad
from scipy.special import j0, j1, jv

import lumenairy.propagators.asm as _asm
import lumenairy.propagators.fft_infra as _fi
from lumenairy.backend.scipy import jv as backend_jv
from lumenairy.propagators.asm import (
    _build_asm_H_square,
    _get_asm_H_natural,
    angular_spectrum_propagate,
)
from lumenairy.propagators.fft_infra import clear_asm_caches
from lumenairy.propagators.fresnel import fresnel_propagate
from lumenairy.propagators.hf import (
    propagate_huygens_fresnel_freespace,
    propagate_huygens_fresnel_with_opl_callable,
)
from lumenairy.propagators.hfpi import (
    PathBundle,
    _spawn_rng,
    accumulate_to_grid,
    apply_aperture_diffraction,
    init_paths_from_field,
    init_paths_stratified,
    propagate_hfpi_freespace_aperture,
    propagate_to_plane,
)
from lumenairy.propagators.mhs import (
    HuygensSurface,
    MhsPipeline,
    asm_subdomain,
)
from lumenairy.propagators.rs import (
    _rs_alias_free_distance,
    rayleigh_sommerfeld_propagate,
)
from lumenairy.propagators.sas import scalable_angular_spectrum_propagate
from lumenairy.propagators.vector_diffraction import richards_wolf_focus
from lumenairy.propagators.vectorial_hfpi import (
    _rigid_rotate,
    propagate_vector_hfpi_freespace_aperture,
)

LAM = 633e-9


# ===========================================================================
# Shared oracles
# ===========================================================================

def _hankel_gaussian(R, z, w0, lam=LAM, nf=4000):
    """EXACT propagated field of ``exp(-(r/w0)**2)`` at distance ``z``.

    Circularly symmetric, so the angular-spectrum integral collapses to a
    Hankel transform pair evaluated by Gauss-Legendre over ``f``::

        E(r, z) = 2 pi Int_0^inf [pi w0^2 exp(-(pi w0 f)^2)]
                               exp(i k z sqrt(1 - (lam f)^2)) J0(2 pi r f) f df

    No FFT, no grid, no library call -- the oracle is independent of
    everything under test.  Its own error floor is the Gauss-Legendre
    truncation: the Gaussian spectrum has fallen to ``exp(-64) = 1.6e-28``
    at the ``f_max = 8/(pi w0)`` cut, and ``nf = 4000`` nodes resolve the
    ``J0`` oscillation to ~1e-12, so the floor is ~1e-12 relative -- four
    decades below the tightest bar asserted against it (6.1e-8).
    """
    k = 2 * np.pi / lam
    ru = np.unique(np.round(np.asarray(R).ravel(), 12))
    inv = np.searchsorted(ru, np.round(np.asarray(R).ravel(), 12))
    fmax = min(8.0 / (np.pi * w0), 0.999999 / lam)
    fn, fw = np.polynomial.legendre.leggauss(nf)
    fv = 0.5 * fmax * (fn + 1.0)
    fwt = 0.5 * fmax * fw
    wgt = (2 * np.pi * np.pi * w0 ** 2 * np.exp(-(np.pi * w0 * fv) ** 2)
           * np.exp(1j * k * z * np.sqrt(np.maximum(1 - (lam * fv) ** 2, 0.0)))
           * fv * fwt)
    out = np.empty(ru.shape, dtype=complex)
    for i0 in range(0, ru.size, 400):
        out[i0:i0 + 400] = j0(2 * np.pi * np.outer(ru[i0:i0 + 400], fv)) @ wgt
    return out[inv].reshape(np.shape(R))


def _gauss_grid(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.hypot(X, Y)
    return np.exp(-(R / w0) ** 2).astype(np.complex128), R


def _rw_ez_over_ex_oracle(x, NA, f, lam=LAM):
    """``E_z / E_x`` on the focal +x axis for an x-polarised aplanatic
    focus, from the radial Debye-Wolf integrals.

    Derived from first principles, NOT from the module under test: the
    ray through the exit pupil at azimuth ``phi`` converges along
    ``s = (-sin t cos p, -sin t sin p, cos t)`` and its polarisation
    rides the rigid rotation ``rho_hat -> e_theta = (cos t cos p,
    cos t sin p, +sin t)``.  Reducing the resulting strength vector on the
    focal ``x`` axis gives

        E_z/E_x = -2i I01 / (I00 + I02)

    with ``I00 = Int sqrt(c) s (1+c) J0(k x s) dt``,
    ``I01 = Int sqrt(c) s^2 J1(k x s) dt``,
    ``I02 = Int sqrt(c) s (1-c) J2(k x s) dt``.  Negative-imaginary for
    small ``x > 0``.  ``scipy.integrate.quad`` at ``limit=400`` returns
    these to ~1e-10 relative.
    """
    k = 2 * np.pi / lam
    tmax = np.arcsin(NA)

    def _i(fn):
        return quad(fn, 0.0, tmax, limit=400)[0]

    i00 = _i(lambda t: np.sqrt(np.cos(t)) * np.sin(t) * (1 + np.cos(t))
             * j0(k * x * np.sin(t)))
    i01 = _i(lambda t: np.sqrt(np.cos(t)) * np.sin(t) ** 2
             * j1(k * x * np.sin(t)))
    i02 = _i(lambda t: np.sqrt(np.cos(t)) * np.sin(t) * (1 - np.cos(t))
             * jv(2, k * x * np.sin(t)))
    return -2j * i01 / (i00 + i02)


# ===========================================================================
# K16 / K10 -- Richards-Wolf E_z sign and the pupil coordinate convention
# ===========================================================================

class TestK16RichardsWolfEzSign:
    """P0: ``richards_wolf_focus`` returned ``E_z`` with the sign opposite
    to ``E_x`` / ``E_y``, on every call.  ``|E_z|^2`` is blind to it, so
    ``debye_wolf_psf``, the GUI dock and every pre-existing test were
    blind too -- no test pinned the sign.
    """

    NA, F, NP = 0.5, 4e-3, 256

    def _focus(self):
        dx_p = 2.0 * self.F * self.NA / self.NP * 1.05
        pupil = np.ones((self.NP, self.NP), dtype=complex)
        return richards_wolf_focus(pupil, LAM, self.NA, self.F, dx_p,
                                   polarization='x')

    def test_ez_over_ex_matches_the_independent_oracle_in_sign(self):
        """code / oracle must be +1, not -1.

        Bar: ``|code/oracle - 1| < 0.10`` per probe point.
        Derivation.  The oracle's own floor is ~1e-10 (``quad`` at
        limit=400).  The code evaluates the same integral by FFT over a
        Cartesian pupil, whose discretisation residual is the ONLY
        expected difference; the orchestrator measured it at 0.2-1.5 % on
        this fixture, and the same residual appears identically on
        ``E_x`` (which is unaffected by the finding).  0.10 sits 1.5-2
        decades above that residual and 20 decades above the oracle
        floor, while the defect being pinned is a SIGN: pre-fix the ratio
        was -1.0006 / -0.9954 / -0.9994 / -0.9670 at the four probes, so
        the failing distance is ~2.0 -- a factor of 20 above the bar.
        """
        Ex, Ey, Ez, xf, _ = self._focus()
        c = self.NP // 2
        for kk in (1, 2, 3, 5):
            x = float(xf[c + kk])
            ratio = complex(Ez[c, c + kk] / Ex[c, c + kk]
                            / _rw_ez_over_ex_oracle(x, self.NA, self.F))
            assert abs(ratio - 1.0) < 0.10, (
                f"richards_wolf_focus E_z/E_x at x_f = {x * 1e6:.3f} um is "
                f"{ratio:+.4f} x the independent Debye-Wolf oracle; the "
                f"pre-fix code gave -1.00 (a flipped sign).")

    def test_im_ez_over_ex_is_negative_for_positive_x(self):
        """The build-free pin the audit asked for.

        Novotny & Hecht eq. 3.66 for an x-polarised aplanatic focus gives
        ``E_z/E_x = -2i I01 cos(phi)/I00``, i.e. NEGATIVE-imaginary just
        off axis on the ``+x`` side.  This is a DECISION (a sign), not a
        reading, so the only bar needed is that the magnitude is far from
        zero: ``|Im(E_z/E_x)| > 0.05`` at the probe, where the measured
        value is 0.697 -- one decade of margin -- and the real part is
        ~1e-17, i.e. the ratio is purely imaginary as the closed form
        requires.
        """
        Ex, _Ey, Ez, xf, _ = self._focus()
        c = self.NP // 2
        r = complex(Ez[c, c + 1] / Ex[c, c + 1])
        assert abs(r.imag) > 0.05, (
            f"|Im(E_z/E_x)| = {abs(r.imag):.4f} is too small for this "
            f"pin to mean anything; fixture broken.")
        assert r.imag < 0.0, (
            f"Im(E_z/E_x) = {r.imag:+.6f} at x_f > 0 on an x-polarised "
            f"focus, but Novotny-Hecht eq. 3.66 requires it NEGATIVE "
            f"(measured +0.6966 pre-fix, -0.6966 post-fix).")
        # ... and POSITIVE on the -x side (E_z is odd in x).
        r_minus = complex(Ez[c, c - 1] / Ex[c, c - 1])
        assert r_minus.imag > 0.0

    def test_ez_is_odd_in_x_even_symmetric_ex_and_zero_on_the_y_axis(self):
        """Symmetry, which the sign fix must not disturb.

        Bars are FFT round-off, not physics: the grid is symmetric, so
        ``E_z(+x) + E_z(-x)`` and ``E_x(+x) - E_x(-x)`` cancel to machine
        precision.  Measured 4.5e-16 and 2.4e-16 relative; the bar 1e-12
        is 3.5 decades above and ~12 decades below any real asymmetry.
        ``E_z`` on the y-axis is 3.1e-19 of its own maximum (exactly zero
        by the ``cos(phi)`` factor); bar 1e-12.
        """
        Ex, _Ey, Ez, _xf, _yf = self._focus()
        c = self.NP // 2
        odd = abs(Ez[c, c + 3] + Ez[c, c - 3]) / abs(Ez[c, c + 3])
        even = abs(Ex[c, c + 3] - Ex[c, c - 3]) / abs(Ex[c, c + 3])
        on_axis = abs(Ez[c + 3, c]) / np.abs(Ez).max()
        assert odd < 1e-12, f"E_z not odd in x: {odd:.3e}"
        assert even < 1e-12, f"E_x not even in x: {even:.3e}"
        assert on_axis < 1e-12, f"E_z not zero on the y-axis: {on_axis:.3e}"

    def test_k10_pupil_is_indexed_by_the_aperture_coordinate(self):
        """K10: a decentred sub-aperture / tilted pupil pins the sign of
        the focal phase ramp, and hence which coordinate ``pupil`` is in.

        A pupil ramp ``exp(+2j pi u x_p)`` moves the focus to
        ``x_f = +u lambda f`` under the APERTURE-coordinate convention
        (the ray leaving aperture point ``x_p`` travels with transverse
        direction ``-x_p/f + u lambda`` and lands at ``+u lambda f``
        independent of ``x_p``).  Under the ray-direction convention it
        would move to ``-u lambda f`` -- the two differ by a point
        inversion, which is invisible for any 180-degree-symmetric pupil
        and is exactly why this needs a decentred fixture.

        Bar: the intensity centroid must land within 15 % of
        ``+u lambda f``, and on the correct SIDE.  Measured +2.3600 um
        against a predicted +2.3737 um (0.6 %, the residual being
        centroid-vs-peak on a finite window), so the bar has 1.2 decades
        of margin, while the wrong convention would land at -2.37 um --
        a failing distance of 2.0 in units of the prediction.
        """
        NA, f, Np = 0.2, 4e-3, 256
        dx_p = 2.0 * f * NA / Np * 1.05
        x_p = (np.arange(Np) - Np / 2) * dx_p
        Xp, Yp = np.meshgrid(x_p, x_p)
        u = 2.3737e-6 / (LAM * f)
        pupil = (np.exp(2j * np.pi * u * Xp)
                 * (np.hypot(Xp, Yp) <= f * NA)).astype(complex)
        Ex, _Ey, _Ez, xf, yf = richards_wolf_focus(
            pupil, LAM, NA, f, dx_p, polarization='x')
        I = np.abs(Ex) ** 2
        cx = float((I.sum(axis=0) * xf).sum() / I.sum())
        pred = u * LAM * f
        assert cx > 0.0, (
            f"focal centroid {cx * 1e6:+.4f} um is on the wrong side of "
            f"the axis for a +u pupil ramp: ``pupil`` is documented as "
            f"indexed by the physical EXIT-PUPIL coordinate (K10).")
        assert abs(cx / pred - 1.0) < 0.15, (
            f"focal centroid {cx * 1e6:+.4f} um vs predicted "
            f"{pred * 1e6:+.4f} um.")


# ===========================================================================
# K9 -- the Rayleigh-Sommerfeld near-field kernel
# ===========================================================================

class TestK9RayleighSommerfeldNearField:
    """P0: the point-sampled RS Green's function aliases for
    ``z < 2 N dx^2 / lambda`` and CREATES energy, with all-default
    arguments, and ``bandlimit=True`` is all-pass in exactly that regime.
    """

    # (N, dx, z) triples the audit measured as failing, with the pre-fix
    # P_out/P_in and relative L2 against the Hankel oracle.
    FAILING = [
        (64, 2e-6, 50e-6, 21.44, 4.50),
        (128, 1e-6, 50e-6, 5.31, 2.08),
        (128, 2e-6, 50e-6, 25.70, 4.95),
    ]

    @pytest.mark.parametrize('N,dx,z,pre_power,pre_rel', FAILING)
    def test_energy_is_conserved_where_the_spatial_kernel_created_it(
            self, N, dx, z, pre_power, pre_rel):
        """Bar: ``|P_out/P_in - 1| < 1e-4``.

        Derivation.  The transfer kernel is exact for a band-limited
        input, so the only loss is the evanescent cut plus FFT round-off;
        measured ``P_out/P_in = 1.000000`` on all three grids (to the six
        digits printed).  The Gaussian's spectrum at the grid Nyquist is
        ``exp(-(pi w0 f_N)^2)`` with ``w0 = 6 um``: 1e-28 at dx = 2 um,
        so nothing real is cut.  1e-4 sits ~4 decades above round-off and
        4-5 decades below the pre-fix values (21.44 / 5.31 / 25.70), i.e.
        a failing distance of 4e4 or more.
        """
        E0, _R = _gauss_grid(N, dx, 6e-6)
        out = rayleigh_sommerfeld_propagate(E0, z, LAM, dx)
        ratio = float(np.sum(np.abs(out) ** 2) / np.sum(np.abs(E0) ** 2))
        assert abs(ratio - 1.0) < 1e-4, (
            f"P_out/P_in = {ratio:.6f} at N={N}, dx={dx * 1e6:.0f} um, "
            f"z={z * 1e6:.0f} um (pre-fix {pre_power}).")

    @pytest.mark.parametrize('N,dx,z,pre_power,pre_rel', FAILING)
    def test_field_matches_the_exact_hankel_oracle(
            self, N, dx, z, pre_power, pre_rel):
        """Bar: relative L2 vs the Hankel oracle ``< 1e-5``.

        Derivation.  The oracle's own floor is ~1e-12 (see
        ``_hankel_gaussian``).  The residual is the Gaussian's truncation
        at the grid edge, measured 5.3e-8 / 6.1e-8 / 5.3e-8 on these
        grids -- identical to what band-limited ASM achieves on the same
        grids, so it is the grid's floor, not the kernel's.  1e-5 sits
        2.2 decades above that and 5.3 decades below the pre-fix values
        (4.50 / 2.08 / 4.95).
        """
        E0, R = _gauss_grid(N, dx, 6e-6)
        ref = _hankel_gaussian(R, z, 6e-6)
        out = rayleigh_sommerfeld_propagate(E0, z, LAM, dx)
        rel = float(np.linalg.norm(out - ref) / np.linalg.norm(ref))
        assert rel < 1e-5, (
            f"relative L2 vs the exact Hankel oracle = {rel:.3e} at N={N}, "
            f"dx={dx * 1e6:.0f} um, z={z * 1e6:.0f} um "
            f"(pre-fix {pre_rel}).")

    @pytest.mark.parametrize('N,dx,z,_p,_r', FAILING)
    def test_the_spatial_kernel_refuses_instead_of_aliasing(
            self, N, dx, z, _p, _r):
        """Forcing the historical kernel inside its alias regime must
        RAISE, naming the threshold and the remedy -- never return the
        silently-wrong field."""
        E0, _R = _gauss_grid(N, dx, 6e-6)
        assert z < _rs_alias_free_distance(N, dx, LAM)
        with pytest.raises(ValueError, match='ALIASES'):
            rayleigh_sommerfeld_propagate(E0, z, LAM, dx, kernel='spatial')

    @pytest.mark.parametrize('N,dx,z', [
        (128, 1e-6, 500e-6), (128, 1e-6, 1e-3), (64, 2e-6, 2e-3),
        (128, 1e-6, 3e-3),
    ])
    def test_far_field_is_bit_identical_to_the_historical_kernel(
            self, N, dx, z):
        """The routed default must not move the regime the audit verified
        as correct.  Above ``2 N dx^2 / lambda`` the default IS the
        historical spatial kernel, so the bar is BIT-identity -- no
        tolerance to derive."""
        E0, _R = _gauss_grid(N, dx, 6e-6)
        assert z >= _rs_alias_free_distance(N, dx, LAM)
        auto = rayleigh_sommerfeld_propagate(E0, z, LAM, dx)
        spatial = rayleigh_sommerfeld_propagate(E0, z, LAM, dx,
                                                kernel='spatial')
        assert auto.tobytes() == spatial.tobytes()

    def test_bandlimit_is_not_a_near_field_remedy(self):
        """``bandlimit=True`` must not be sold as the fix: on the padded
        grid it can only DISCARD content.

        Bar: ``rel(bandlimit=True) > 100 x rel(bandlimit=False)`` at
        z = 3 mm, where measured values are 1.80e-2 and 4.42e-8 -- a
        ratio of 4e5, i.e. 3.6 decades above the bar.  This pins the
        DIRECTION of the effect (the docstring's claim), not a value.
        """
        N, dx, z = 128, 1e-6, 3e-3
        E0, R = _gauss_grid(N, dx, 6e-6)
        ref = _hankel_gaussian(R, z, 6e-6)
        nn = np.linalg.norm(ref)
        r_false = float(np.linalg.norm(
            rayleigh_sommerfeld_propagate(E0, z, LAM, dx, bandlimit=False)
            - ref) / nn)
        r_true = float(np.linalg.norm(
            rayleigh_sommerfeld_propagate(E0, z, LAM, dx, bandlimit=True)
            - ref) / nn)
        assert r_true > 100.0 * r_false, (
            f"bandlimit=True {r_true:.3e} vs False {r_false:.3e}: the "
            f"docstring's 'bandlimit is not a near-field remedy' claim "
            f"is what this pins.")

    def test_alias_threshold_formula(self):
        """``2 N dx^2 / lambda`` -- an algebraic identity, exact."""
        assert _rs_alias_free_distance(64, 2e-6, LAM) == pytest.approx(
            2 * 64 * (2e-6) ** 2 / LAM, rel=1e-15)

    def test_invalid_kernel_token_raises_with_the_fn_name_prefix(self):
        with pytest.raises(ValueError,
                           match='rayleigh_sommerfeld_propagate: kernel'):
            rayleigh_sommerfeld_propagate(
                np.ones((8, 8), complex), 1e-3, LAM, 1e-6, kernel='nope')


# ===========================================================================
# K1 -- the single-FFT Fresnel chirp-sampling guard
# ===========================================================================

class TestK1FresnelChirpGuard:
    """P1: ``fresnel_propagate`` had no chirp-sampling guard -- 33 %
    relative error at ``z = 0.25 z_crit`` with zero warnings."""

    N, DX = 128, 2e-6

    def _run(self, q):
        z_crit = self.N * self.DX ** 2 / LAM
        x = (np.arange(self.N) - self.N / 2) * self.DX
        X, Y = np.meshgrid(x, x)
        E = np.exp(-(X ** 2 + Y ** 2) / (40e-6) ** 2).astype(complex)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            fresnel_propagate(E, q * z_crit, LAM, self.DX)
        return [r for r in rec
                if issubclass(r.category, RuntimeWarning)
                and 'UNDER-SAMPLED' in str(r.message)]

    @pytest.mark.parametrize('q,pre_err', [(0.25, 3.34e-1), (0.5, 1.29e-4)])
    def test_warns_below_the_validity_bound(self, q, pre_err):
        """Decision, not a reading: a warning either fires or it does
        not.  The audit measured 0 warnings at both of these and
        relative errors of 3.34e-1 / 1.29e-4 against an 8x-oversampled
        direct Fresnel quadrature."""
        assert self._run(q), (
            f"no sampling warning at z = {q} z_crit, where the measured "
            f"relative error is {pre_err}.")

    @pytest.mark.parametrize('q', [1.0, 2.0, 4.0])
    def test_silent_at_and_above_the_bound(self, q):
        """Counter-pin: the guard must not cry wolf where the kernel is
        valid (measured relative error 4.0e-6 / 1.2e-6 / 4.6e-7)."""
        assert not self._run(q)


# ===========================================================================
# K2 -- backend.scipy.jv argument order
# ===========================================================================

class TestK2BackendBesselArgumentOrder:
    """P1: ``backend.scipy.jv(v, x)`` called ``scipy.special.jv(x, v)``."""

    @pytest.mark.parametrize('v,x,pre_fix', [
        (0, 2.0, 0.0), (1, 3.0, 0.019563354),
        (2, 1.5, 0.491293779), (0.5, 4.0, 0.000160736),
    ])
    def test_scalar_values_match_scipy(self, v, x, pre_fix):
        """Oracle: ``scipy.special.jv`` itself.  Bar: 1e-14 relative --
        the two calls are the SAME routine with the arguments in the
        right order, so any difference beyond round-off is the bug.  The
        pre-fix values are the transposed evaluations listed above; the
        smallest failing distance is |0.4913 - 0.2321| = 0.26, 12 decades
        above the bar."""
        got = float(backend_jv(v, x))
        ref = float(_sp.jv(v, x))
        assert got == pytest.approx(ref, rel=1e-14, abs=1e-15), (
            f"backend jv({v}, {x}) = {got:+.9f}, scipy = {ref:+.9f} "
            f"(pre-fix {pre_fix:+.9f}).")

    def test_array_argument(self):
        """The array case whose FIRST element was right by coincidence
        (``J1(1)``) -- exactly what a smoke test would have checked."""
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(np.asarray(backend_jv(1, x)),
                                   _sp.jv(1, x), rtol=1e-14)


# ===========================================================================
# K3 -- SAS float32 cancellation
# ===========================================================================

class TestK3SasSinglePrecisionCancellation:
    """P1: SAS formed ``h_AS - h_Fr`` (a near-1 cancellation) in float32
    and multiplied it by ``k z`` -- 0.90 rad of phase error at z = 1 m,
    and long distance is SAS's reason to exist."""

    def _phase_err(self, z):
        N, dx = 128, 1e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        E64 = np.exp(-(X ** 2 + Y ** 2) / (20e-6) ** 2).astype(np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clear_asm_caches()
            a = scalable_angular_spectrum_propagate(E64, z, LAM, dx)[0]
            clear_asm_caches()
            b = scalable_angular_spectrum_propagate(
                E64.astype(np.complex64), z, LAM, dx)[0]
        m = np.abs(a) > 1e-3 * np.abs(a).max()
        ph = np.angle(np.exp(1j * np.angle(
            b[m].astype(np.complex128) / a[m])))
        return float(np.max(np.abs(ph)))

    def test_single_precision_error_does_not_grow_with_kz(self):
        """The library's own complex64 contract (``fft_infra`` 190-195)
        says single precision stays at the FFT noise floor and does NOT
        degrade with phase magnitude.  Pin exactly that, as a RATIO
        between two distances 300x apart, so no absolute noise floor has
        to be pinned:

            err(z = 1 m) / err(z = 3.24 mm)  <  10

        Measured post-fix 6.8e-6 / 1.5e-5 = 0.45 -- 1.3 decades below the
        bar.  Pre-fix the kernel-level error was 2.9e-3 rad at 3.24 mm
        and 0.902 rad at 1 m, i.e. a ratio of 309 -- 1.5 decades ABOVE
        it.  The bar has a decade of gap on each side.
        """
        near = self._phase_err(3.24e-3)
        far = self._phase_err(1.0)
        assert far / max(near, 1e-30) < 10.0, (
            f"complex64 SAS phase error grows with k z: {near:.3e} rad at "
            f"z = 3.24 mm -> {far:.3e} rad at z = 1 m (ratio "
            f"{far / near:.1f}; pre-fix 309).")

    def test_single_precision_error_is_at_the_float32_floor(self):
        """Absolute companion: float32 carries ~1.2e-7 relative, so a
        phase error of a few times 1e-5 rad IS the floor for this grid.
        Bar 1e-3 rad -- 1.8 decades above the measured 1.5e-5 and 2.9
        decades below the pre-fix 0.902 rad at z = 1 m."""
        assert self._phase_err(1.0) < 1e-3


# ===========================================================================
# K5 / K8 -- ASM kernel workspace and the bit-exactness contracts
# ===========================================================================

class TestK5AsmKernelWorkspaceCap:

    def test_capped_build_is_byte_identical_to_the_whole_grid_build(self):
        """The band width is a free choice -- H is elementwise in
        (row, column) -- so the bar is BIT-identity, no tolerance."""
        N, dx, z = 512, 1e-6, 1e-3
        saved = _asm._ASM_H_BUILD_BAND_ELEMS
        try:
            clear_asm_caches()
            _asm._ASM_H_BUILD_BAND_ELEMS = 1 << 40      # whole grid
            whole = np.array(_get_asm_H_natural(
                N, N, dx, dx, LAM, z, True, np.complex128, np), copy=True)
            clear_asm_caches()
            _asm._ASM_H_BUILD_BAND_ELEMS = saved        # shipped cap
            capped = np.array(_get_asm_H_natural(
                N, N, dx, dx, LAM, z, True, np.complex128, np), copy=True)
        finally:
            _asm._ASM_H_BUILD_BAND_ELEMS = saved
            clear_asm_caches()
        assert whole.tobytes() == capped.tobytes()

    def test_the_cap_actually_bites_at_a_realistic_grid(self):
        """Counter-pin: a cap that never engages fixes nothing.  At
        N = 2048 the shipped 2**18-element band is 128 rows, so the
        workspace is 1/16 of the grid (tracemalloc: 4.06 -> 1.26 full
        grids of transient, byte-identical)."""
        rows = _asm._ASM_H_BUILD_BAND_ELEMS // 2048
        assert 1 <= rows < 2048, rows


class TestK8BitExactnessAndCacheHygiene:

    @pytest.mark.parametrize('N,dx,bl', [
        (64, 1e-6, True), (255, 0.5e-6, False), (127, 0.3e-6, False),
        (129, 0.7e-6, False), (256, 0.5e-6, True),
    ])
    def test_square_builder_is_bit_exact_to_the_natural_builder(
            self, N, dx, bl):
        """``_build_asm_H_square``'s documented "bit-exact" contract.
        Pre-fix it formed the frequency axis by DIVISION while the shared
        cache multiplies by the reciprocal -- up to 1 ULP apart whenever
        ``1/(N dx)`` is inexact.  Measured max|dH| = 9.096e-13 at
        N = 255 / dx = 0.5 um / bandlimit=False, now 0."""
        clear_asm_caches()
        a = _build_asm_H_square(N, dx, 1e-3, LAM, np.complex128, bl)
        b = np.fft.fftshift(_get_asm_H_natural(
            N, N, dx, dx, LAM, 1e-3, bl, np.complex128, np))
        assert a.tobytes() == b.tobytes()

    def test_cached_transfer_function_is_handed_out_read_only(self):
        """The "callers must not mutate it in place" convention becomes
        an enforced invariant: two lookups at one key share memory, and
        the shared array refuses writes."""
        clear_asm_caches()
        H1 = _get_asm_H_natural(128, 128, 1e-6, 1e-6, LAM, 1e-3, True,
                                np.complex128, np)
        H2 = _get_asm_H_natural(128, 128, 1e-6, 1e-6, LAM, 1e-3, True,
                                np.complex128, np)
        assert np.shares_memory(H1, H2)
        assert not H1.flags.writeable
        with pytest.raises(ValueError):
            H1[0, 0] = 0.0

    def test_returned_transfer_function_stays_writeable(self):
        """Counter-pin: the PUBLIC return copies, so a caller doing
        ``E, H = asm(..., return_transfer_function=True); H *= mask``
        keeps working."""
        E = np.ones((128, 128), complex)
        _out, H = angular_spectrum_propagate(
            E, 1e-3, LAM, 1e-6, return_transfer_function=True)
        assert H.flags.writeable
        H *= 2.0

    def test_propagation_result_accepts_the_numpy2_copy_keyword(self):
        """numpy 2 passes ``copy=`` to ``__array__``; pre-fix
        ``np.array(result, copy=True)`` emitted a DeprecationWarning and
        ``copy=False`` raised ValueError."""
        from lumenairy.propagators.dispatch import propagate
        r = propagate(np.ones((32, 32), complex), z=1e-3, wavelength=LAM,
                      dx=1e-6, method='asm')
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            a = np.array(r, copy=True)
            b = np.array(r, copy=False)
        assert a.shape == (32, 32) and b.shape == (32, 32)


# ===========================================================================
# K4 / K7 -- pyFFTW plan slots and the failure blacklist
# ===========================================================================

class TestK4PerSlotPlanLocks:

    def test_ping_pong_slots_do_not_share_a_lock(self):
        """P2: one lock per ENTRY serialised the two ping-pong slots, so
        the double buffer delivered zero concurrency (measured max
        simultaneous threads inside the critical section = 1 with 4
        threads).  Structural pin: distinct buffers must come with
        distinct locks."""
        if not _fi.PYFFTW_AVAILABLE:
            # Not a resource skip: with no pyFFTW there are no plan
            # entries at all, so the invariant is vacuously held and
            # there is nothing this test could read.
            return
        shape = (128, 128)
        dt = np.dtype(np.complex128)
        _fi._fft2(np.ones(shape, dtype=dt))
        p1, b1, l1, nb = _fi._get_or_make_plan('fwd', shape, dt,
                                               _fi.FFTW_THREADS)
        p2, b2, l2, _ = _fi._get_or_make_plan('fwd', shape, dt,
                                              _fi.FFTW_THREADS)
        if nb < 2:
            return          # single-buffer mode: one slot, one lock
        assert b1 is not b2, 'ping-pong slots share a buffer'
        assert l1 is not l2, (
            'ping-pong slots share ONE lock, so two threads at the same '
            'key serialise even though they hold different plans and '
            'different buffers (measured concurrency 1 of 2).')
        assert isinstance(l1, type(threading.Lock()))


class TestK7PyfftwBlacklistKey:

    def test_one_dtype_failure_does_not_blacklist_the_others(self):
        """P2: keyed on the bare SHAPE, one complex128 MemoryError at
        (512, 512) also skipped complex64 (half the memory) and the
        inverse direction (its own plan and buffer)."""
        saved = set(_fi._PYFFTW_BAD_SHAPES)
        try:
            _fi._PYFFTW_BAD_SHAPES.clear()
            a = np.ones((512, 512), dtype=np.complex128)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _fi._handle_pyfftw_failure(a, 'fft2', MemoryError('probe'))
            key = _fi._pyfftw_bad_key
            assert key((512, 512), np.complex128, 'fwd') in _fi._PYFFTW_BAD_SHAPES
            for dt, dirn in ((np.complex64, 'fwd'), (np.complex128, 'inv'),
                             (np.complex64, 'inv')):
                assert key((512, 512), dt, dirn) not in _fi._PYFFTW_BAD_SHAPES, (
                    f'a complex128 forward failure blacklisted '
                    f'{np.dtype(dt).str} / {dirn}')
        finally:
            _fi._PYFFTW_BAD_SHAPES.clear()
            _fi._PYFFTW_BAD_SHAPES.update(saved)


# ===========================================================================
# K11 / K21 / K20 -- hf free-space resample, pitch gate, return type
# ===========================================================================

class TestK11ResampleDoesNotFabricateEnergy:

    def test_cropped_window_carries_only_the_power_inside_it(self):
        """P1: the ``sqrt(p_in/p_out)`` Parseval renormalisation restored
        the FULL source power after a resample that physically CROPPED
        the field -- measured a +-16 um window genuinely holding 67.27 %
        of the power returned carrying 100.00 % (amplitude x1.219).

        Oracle: the power of the returned (un-renormalised) field cannot
        exceed the power the source grid holds inside the requested
        window.  Bar: returned / true-inside-window within 2 %, where the
        2 % is the bicubic interpolation drift the renormalisation is
        legitimately there to correct.  Pre-fix the same ratio was
        1/0.67 = 1.49 -- 1.5 decades above the bar.
        """
        N, dx = 64, 2e-6
        E, _R = _gauss_grid(N, dx, 20e-6)
        dx_out = 0.5e-6
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            E_out, dx_ret = propagate_huygens_fresnel_freespace(
                E, 1e-3, LAM, dx, output_dx=dx_out)
        assert dx_ret == pytest.approx(dx_out, rel=1e-15)
        assert [r for r in rec if issubclass(r.category, RuntimeWarning)
                and 'CROPS' in str(r.message)], 'the crop was not reported'

        native = rayleigh_sommerfeld_propagate(E, 1e-3, LAM, dx)
        x = (np.arange(N) - N / 2) * dx
        half = 0.5 * N * dx_out
        p_full = float(np.sum(np.abs(native) ** 2)) * dx ** 2
        # The source pitch (2 um) is four times the target pitch, so the
        # window edge falls INSIDE a source pixel: bracket the truth
        # between "every pixel wholly inside" and "every pixel that the
        # window touches" rather than pinning a boundary convention.
        strict = np.abs(x) <= half - dx
        loose = np.abs(x) <= half + dx
        p_lo = float(np.sum(np.abs(native[np.ix_(strict, strict)]) ** 2)) * dx ** 2
        p_hi = float(np.sum(np.abs(native[np.ix_(loose, loose)]) ** 2)) * dx ** 2
        p_ret = float(np.sum(np.abs(E_out) ** 2)) * dx_out ** 2
        assert p_lo <= p_ret <= p_hi, (
            f'returned power {p_ret:.6e} is outside the bracket '
            f'[{p_lo:.6e}, {p_hi:.6e}] the requested window can hold; '
            f'pre-fix the renormalisation restored the FULL source power '
            f'{p_full:.6e}.')
        # The finding itself: the returned field must NOT carry the whole
        # source power.  Bar: < 0.95 of it, where the window here holds
        # {p_hi/p_full:.2f} at most and the pre-fix value was exactly 1.00
        # (the audit measured 67.27 % held vs 100.00 % returned).
        assert p_ret < 0.95 * p_full, (
            f'returned/full = {p_ret / p_full:.4f}: energy is still being '
            f'fabricated across the crop.')

    def test_uncropped_resample_is_unchanged(self):
        """Counter-pin: when the target window covers the source, the
        restriction is the identity, so the interpolation-drift
        correction still applies and no warning fires."""
        N, dx = 64, 2e-6
        E, _R = _gauss_grid(N, dx, 8e-6)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out, _dx = propagate_huygens_fresnel_freespace(
                E, 1e-3, LAM, dx, output_dx=4e-6)
        assert not [r for r in rec if issubclass(r.category, RuntimeWarning)
                    and 'CROPS' in str(r.message)]
        assert np.isfinite(out).all()


class TestK21PitchGate:

    @pytest.mark.parametrize('dx,dx_out', [(1e-6, 1.005e-6), (1e-7, 1.1e-7)])
    def test_a_real_pitch_change_is_not_a_silent_no_op(self, dx, dx_out):
        """P2: the gate was ``np.isclose(dx, target_dx, rtol=1e-12)``,
        which still carries numpy's default ``atol = 1e-8`` -- 10 nm in
        this library's METRES.  A 0.5 % change at 1 um and a 10 % change
        at 100 nm both compared equal, so the UN-resampled field came
        back labelled with the requested pitch."""
        N = 32
        E, _R = _gauss_grid(N, dx, 6 * dx)
        native = rayleigh_sommerfeld_propagate(E, 1e-3, LAM, dx)
        out, ret = propagate_huygens_fresnel_freespace(
            E, 1e-3, LAM, dx, output_dx=dx_out)
        assert ret == pytest.approx(dx_out, rel=1e-15)
        assert not np.array_equal(out, native), (
            f'dx {dx:.3e} -> {dx_out:.3e} returned the un-resampled field '
            f'labelled with the requested pitch.')

    def test_an_exact_no_op_still_short_circuits_bitwise(self):
        """Counter-pin: the short-circuit must still fire for a genuine
        no-op, bit-for-bit."""
        N, dx = 32, 1e-6
        E, _R = _gauss_grid(N, dx, 6e-6)
        native = rayleigh_sommerfeld_propagate(E, 1e-3, LAM, dx)
        out, ret = propagate_huygens_fresnel_freespace(
            E, 1e-3, LAM, dx, output_dx=dx)
        assert ret == dx
        assert out.tobytes() == native.tobytes()


class TestK20UniformReturnType:

    @pytest.mark.parametrize('method', ['asm', 'gbd', 'hf', 'hfpi'])
    def test_every_output_grid_capable_method_returns_an_ndarray(self, method):
        """P2: ``hf``'s free-space branch was the only one that changed
        its RETURN TYPE when an output grid was requested -- including
        for a request that is a strict no-op -- while the dispatcher's
        own W9-4 message steers callers to it."""
        from lumenairy.propagators.dispatch import propagate
        N, dx = 32, 20e-6
        E, _R = _gauss_grid(N, dx, 6 * dx)
        extra = {}
        if method == 'hfpi':
            # The three-leg free-space form needs its full geometry.
            extra = dict(z_to_aperture=5e-4, aperture_radius=300e-6,
                         z_aperture_to_output=5e-4, n_paths=4000,
                         rng=2, on_undersampled='silent')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bare = propagate(E, z=1e-3, wavelength=LAM, dx=dx,
                             method=method, return_result=False, **extra)
            grid = propagate(E, z=1e-3, wavelength=LAM, dx=dx, method=method,
                             output_grid={'N': 16, 'dx': 40e-6},
                             return_result=False, **extra)
        assert isinstance(bare, np.ndarray), type(bare).__name__
        assert isinstance(grid, np.ndarray), (
            f"method={method!r} with output_grid returned "
            f"{type(grid).__name__}, not ndarray")
        assert grid.shape == (16, 16)


# ===========================================================================
# K22 -- the HF OPL quadrature's output batching
# ===========================================================================

class TestK22HfQuadratureChunking:

    @staticmethod
    def _opl(s1x, s1y, s2x, s2y):
        z, lam = 50e-3, 1e-6
        return np.sqrt((s1x - s2x) ** 2 + (s1y - s2y) ** 2 + z * z) / lam

    @pytest.mark.parametrize('van_vleck', [True, False])
    def test_batching_is_bit_identical_to_the_per_pixel_evaluation(
            self, van_vleck):
        """``chunk_output`` changes only how many output pixels share one
        vectorised ``opl_fn`` call, never the arithmetic.  Bar:
        BIT-identity, no tolerance to derive."""
        N, dx, N_out = 48, 2e-6, 8
        E, _R = _gauss_grid(N, dx, 8 * dx)
        g = (np.arange(N_out) - N_out / 2) * dx
        kw = dict(opl_fn=self._opl, output_grid_x=g, output_grid_y=g,
                  input_grid_dx=dx, apply_van_vleck=van_vleck)
        auto = propagate_huygens_fresnel_with_opl_callable(E, **kw)
        one = propagate_huygens_fresnel_with_opl_callable(
            E, chunk_output=1, **kw)
        many = propagate_huygens_fresnel_with_opl_callable(
            E, chunk_output=N_out * N_out, **kw)
        assert auto.tobytes() == one.tobytes()
        assert auto.tobytes() == many.tobytes()

    def test_a_scalar_only_callable_falls_back_and_still_agrees(self):
        """Backward compatibility: the pre-v5.46 contract said the output
        coordinates are SCALARS.  A callable that cannot take the batched
        form must be detected and served by the per-pixel path -- with a
        diagnostic, and with the same field."""
        N, dx, N_out = 32, 4e-6, 4
        E, _R = _gauss_grid(N, dx, 8 * dx)
        g = (np.arange(N_out) - N_out / 2) * dx

        def scalar_only(s1x, s1y, s2x, s2y):
            return self._opl(s1x, s1y, float(s2x), float(s2y))

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            got = propagate_huygens_fresnel_with_opl_callable(
                E, opl_fn=scalar_only, output_grid_x=g, output_grid_y=g,
                input_grid_dx=dx, apply_van_vleck=True)
        assert [r for r in rec if issubclass(r.category, RuntimeWarning)
                and 'did not broadcast' in str(r.message)]
        ref = propagate_huygens_fresnel_with_opl_callable(
            E, opl_fn=self._opl, output_grid_x=g, output_grid_y=g,
            input_grid_dx=dx, apply_van_vleck=True)
        assert got.tobytes() == ref.tobytes()

    def test_chunk_output_validates_its_argument(self):
        N, dx = 16, 4e-6
        E, _R = _gauss_grid(N, dx, 4 * dx)
        g = (np.arange(4) - 2) * dx
        with pytest.raises(ValueError, match='chunk_output'):
            propagate_huygens_fresnel_with_opl_callable(
                E, opl_fn=self._opl, output_grid_x=g, output_grid_y=g,
                input_grid_dx=dx, chunk_output=0)


# ===========================================================================
# K12 / K13 / K18 / K19 / K23 / K14 -- HFPI
# ===========================================================================

class TestK12WavelengthIsRequired:

    def test_scalar_aperture_requires_wavelength(self):
        """P1: ``wavelength: float = 0.0`` gated the ``1/(i lambda)``
        Kirchhoff prefactor, so omitting it silently dropped the physics
        -- every weight wrong by 1/lambda = 1.58e6 in magnitude AND by
        -90 degrees in phase."""
        E = np.ones((8, 8), complex)
        paths = init_paths_from_field(E, 2e-6, n_paths=64, wavelength=LAM,
                                      rng=1)
        with pytest.raises(TypeError):
            apply_aperture_diffraction(paths, 1e-3)
        with pytest.raises(ValueError, match='wavelength must be a positive'):
            apply_aperture_diffraction(paths, 1e-3, wavelength=0.0)
        with pytest.raises(ValueError):
            apply_aperture_diffraction(paths, 1e-3, wavelength=-1.0)

    def test_the_prefactor_is_the_one_that_was_being_dropped(self):
        """The magnitude of what the default used to skip: the Kirchhoff
        prefactor is ``1/(i lambda)`` -- 1.58e6 in magnitude at 633 nm
        and exactly -90 degrees in phase.

        Read off the re-emission factor directly, on the 'legacy'
        measure so the assertion is about the PREFACTOR and not about the
        v5.46.1 intermediate-leg Jacobian (verify V1).
        """
        E = np.ones((8, 8), complex)
        paths = init_paths_from_field(E, 2e-6, n_paths=64, wavelength=LAM,
                                      rng=1)
        out = apply_aperture_diffraction(paths, 1e-3, wavelength=LAM,
                                         normalisation='legacy')
        ratio = complex(np.asarray(out.weights)[0]
                        / np.asarray(paths.weights)[0])
        # legacy factor = 0.5(cos_in + cos_out) * (1/(i lam)) * Omega/n,
        # all real and positive except the 1/(i lam), so the PHASE is the
        # prefactor's alone.
        assert np.angle(ratio) == pytest.approx(-np.pi / 2, abs=1e-12), (
            f"re-emission phase {np.angle(ratio):+.6f} rad; the "
            f"1/(i lambda) Kirchhoff prefactor contributes exactly "
            f"-pi/2 and nothing else in that factor is complex.")
        assert abs(complex(1.0 / (1j * LAM))) == pytest.approx(
            1.0 / LAM, rel=1e-15)


class TestK18SourceAreaNormalisation:

    @pytest.mark.parametrize('n,pre_ratio', [(1, 1.000004), (4, 0.062140),
                                             (16, 0.003945)])
    def test_source_weights_integrate_to_the_exact_hf_source_term(
            self, n, pre_ratio):
        """P1: the source pixel is drawn uniformly over ``Ny*Nx`` pixels,
        so the unbiased estimate carries the WHOLE illuminated area
        ``Ny*Nx*dx^2``, not one pixel's ``dx^2``.  Pre-fix the ratio to
        the exact integral tracked ``1/N_pix`` -- 4096x low at 64x64.

        Oracle: the closed form of the source term for one unit-amplitude
        pixel emitting into a cone of half-angle ``theta_max``,
        ``Int E cos(theta) dx^2/(i lambda) dOmega
          = (dx^2/(i lambda)) pi sin^2(theta_max)``.

        Bar: ``|ratio - 1| < 0.25``.  Derivation: this is a Monte-Carlo
        sum of 200 000 draws with only ``1/N_pix`` of them landing on the
        single non-zero pixel, so the estimator's own relative sd is
        ``sqrt(N_pix/200000)`` = 0.2 % / 0.9 % / 3.6 % at n = 1 / 4 / 16;
        0.25 is 1-2 decades above that, and 0.6-2.4 decades below the
        pre-fix shortfalls (1 - 0.0621 = 0.94 and 1 - 0.0039 = 0.996).
        """
        E = np.zeros((n, n), dtype=complex)
        E[n // 2, n // 2] = 1.0
        dx, cone = 2e-6, 0.20
        b = init_paths_from_field(E, dx, n_paths=200000, wavelength=LAM,
                                  rng=11, cone_half_angle=cone)
        got = complex(np.sum(np.asarray(b.weights)))
        exact = (1.0 / (1j * LAM)) * dx * dx * np.pi * np.sin(cone) ** 2
        ratio = abs(got) / abs(exact)
        assert abs(ratio - 1.0) < 0.25, (
            f'sum(weights)/exact = {ratio:.6f} at a {n}x{n} source '
            f'(pre-fix {pre_ratio}, tracking 1/N_pix = {1.0 / (n * n):.6f}).')


class TestK13BinningJacobian:

    def test_the_free_space_estimator_reproduces_the_hf_integral(self):
        """P1: the estimator's exact bias law
        ``E[HFPI]/E_true = dx_out^2 cos(theta)/(N_src_px r)`` made the
        returned amplitude depend on the OUTPUT PIXEL AREA and the SOURCE
        PIXEL COUNT -- rebinning the grids changed the answer with no
        physics change, and ``|E|max`` moved 14x between 2 M and 8 M
        paths.

        Oracle: band-limited ASM on the same geometry, which is exact
        there (6.1e-8 relative L2 against the Hankel quadrature).
        Estimator read out as the UNBIASED least-squares complex scale
        ``sum(E_hfpi conj(E_asm))/sum(|E_asm|^2)`` -- the mean of
        ``|E_hfpi|/|E_asm|`` is biased high because ``|sum|^2`` carries
        the estimator's noise power.

        Bar: ``|scale - 1| < 0.15``.  Derivation: at 2 M paths into a
        32x32 grid the Monte-Carlo scatter across seeds and path counts
        is 2-4 % (measured 1.0115 / 0.9758 / 0.9916 / 0.9866), and the
        cone truncation contributes a few more; 0.15 is ~0.6 decades
        above that.  The pre-fix scale on this fixture is
        ``dx^2/(N_pix z)`` = 7.8e-12 -- ELEVEN decades below the bar.
        """
        N, dx, w0, z = 32, 4e-6, 12e-6, 2e-3
        E0, _R = _gauss_grid(N, dx, w0)
        ref = angular_spectrum_propagate(E0, z=z, wavelength=LAM, dx=dx)
        m = np.abs(ref) > 0.05 * np.abs(ref).max()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            paths = init_paths_from_field(E0, dx, n_paths=2_000_000,
                                          wavelength=LAM, rng=1,
                                          cone_half_angle=0.05)
            paths = propagate_to_plane(paths, z_target=z, wavelength=LAM)
            phys = accumulate_to_grid(paths, Ny=N, Nx=N, dx=dx,
                                      on_undersampled='silent')
            legacy = accumulate_to_grid(paths, Ny=N, Nx=N, dx=dx,
                                        on_undersampled='silent',
                                        normalisation='legacy')
        denom = float(np.sum(np.abs(ref[m]) ** 2))
        scale = abs(complex(np.sum(phys[m] * np.conj(ref[m]))) / denom)
        assert abs(scale - 1.0) < 0.15, (
            f'corrected HFPI / ASM least-squares scale = {scale:.4f}; the '
            f'estimator is meant to BE the Huygens-Fresnel integral now.')
        # ... and the legacy path still carries the documented bias law,
        # so the two differ by exactly the Jacobian (r / dx_out^2, with
        # cos(theta) ~ 1 inside a 0.05 rad cone).  Bar 20 %: the
        # remaining scatter is the same MC noise as above.
        scale_legacy = abs(complex(np.sum(legacy[m] * np.conj(ref[m]))) / denom)
        assert abs(scale_legacy / (dx * dx / z) / scale - 1.0) < 0.20

    def test_physical_normalisation_refuses_an_unpropagated_bundle(self):
        """Instead of silently returning zeros when ``r = 0``."""
        n = 64
        pos = np.zeros((n, 3))
        dirs = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
        paths = PathBundle(positions=pos, directions=dirs,
                           weights=np.ones(n, dtype=complex),
                           opl=np.zeros(n), alive=np.ones(n, dtype=bool))
        with pytest.raises(ValueError, match="normalisation='physical'"):
            accumulate_to_grid(paths, Ny=8, Nx=8, dx=1e-6,
                               on_undersampled='silent')
        out = accumulate_to_grid(paths, Ny=8, Nx=8, dx=1e-6,
                                 on_undersampled='silent',
                                 normalisation='legacy')
        assert complex(np.sum(out)) == pytest.approx(float(n), abs=1e-9)


class TestK19RngDefaultDrawsEntropy:

    def test_default_runs_differ(self):
        """P1: every consumer wrote
        ``RandomState(rng if rng is not None else 0)``, so the DEFAULT
        was the fixed seed 0 -- two default runs were byte-identical and
        the canonical re-seed error estimate of a 1/sqrt(N) estimator was
        identically zero."""
        E = np.ones((8, 8), complex)
        a = init_paths_from_field(E, 2e-6, n_paths=512, wavelength=LAM)
        b = init_paths_from_field(E, 2e-6, n_paths=512, wavelength=LAM)
        assert not np.array_equal(np.asarray(a.directions),
                                  np.asarray(b.directions)), (
            'two rng=None runs are byte-identical: the default is still a '
            'fixed seed, so a re-seed error estimate reads exactly zero.')

    def test_default_is_not_seed_zero(self):
        E = np.ones((8, 8), complex)
        a = init_paths_from_field(E, 2e-6, n_paths=512, wavelength=LAM)
        z = init_paths_from_field(E, 2e-6, n_paths=512, wavelength=LAM, rng=0)
        assert not np.array_equal(np.asarray(a.directions),
                                  np.asarray(z.directions))

    def test_an_explicit_seed_is_still_reproducible(self):
        """Counter-pin: seeding must still pin the stream exactly."""
        E = np.ones((8, 8), complex)
        a = init_paths_from_field(E, 2e-6, n_paths=512, wavelength=LAM, rng=7)
        b = init_paths_from_field(E, 2e-6, n_paths=512, wavelength=LAM, rng=7)
        assert np.array_equal(np.asarray(a.directions),
                              np.asarray(b.directions))


class TestK24SpawnRngIsAPureFunction:

    def test_generator_branch_does_not_mutate_the_parent(self):
        """P3: ``rng.spawn(stream_index + 1)[-1]`` advanced the caller's
        generator on every call, so ``stream 1 drawn AFTER stream 0``
        differed from ``stream 1 drawn alone`` -- the stream index was
        not a stable key."""
        def draws(order):
            parent = np.random.default_rng(12345)
            out = {}
            for i in order:
                out[i] = np.asarray(_spawn_rng(parent, i).random(4))
            return out
        first = draws([0, 1])
        alone = draws([1])
        np.testing.assert_array_equal(first[1], alone[1])

    def test_stream_indices_are_independent(self):
        parent = np.random.default_rng(999)
        a = np.asarray(_spawn_rng(parent, 0).random(8))
        parent2 = np.random.default_rng(999)
        b = np.asarray(_spawn_rng(parent2, 1).random(8))
        assert not np.array_equal(a, b)


class TestK14K23GuardsAndCaps:

    def test_cone_half_angle_reaches_the_free_space_entry_points(self):
        """P2: the under-sampling guard's own recommended remedy was a
        ``TypeError`` on both free-space entry points, so the only
        free-space HFPI entry in the library was the one that could not
        take the lever it recommended."""
        E = np.ones((16, 16), complex)
        kw = dict(z_to_aperture=1e-3, aperture_radius=30e-6,
                  z_aperture_to_output=1e-3, wavelength=LAM, n_paths=4096,
                  rng=3, on_undersampled='silent')
        wide = propagate_hfpi_freespace_aperture(
            E, 5e-6, cone_half_angle=np.pi / 2 - 1e-6, **kw)
        narrow = propagate_hfpi_freespace_aperture(
            E, 5e-6, cone_half_angle=0.05, **kw)
        assert np.count_nonzero(narrow) > np.count_nonzero(wide), (
            'narrowing the cone must put more paths on the grid; that is '
            'the whole point of the lever the guard recommends.')
        ex, ey = propagate_vector_hfpi_freespace_aperture(
            E, np.zeros_like(E), 5e-6, cone_half_angle=0.05, **kw)
        assert ex.shape == (16, 16)

    def test_the_vector_accumulator_shares_the_undersampling_guard(self):
        """P2: v4.13.1 forked the vector accumulator for index sharing,
        so the v5.31 guard never ran on it -- identical geometry, scalar
        warned, vector silent (2 of 4096 pixels non-zero)."""
        E = np.ones((64, 64), complex)
        kw = dict(z_to_aperture=2e-3, aperture_radius=30e-6,
                  z_aperture_to_output=2e-3, wavelength=LAM, n_paths=20000,
                  rng=5)
        with pytest.warns(RuntimeWarning, match='UNDER-SAMPLED'):
            propagate_vector_hfpi_freespace_aperture(
                E, np.zeros_like(E), 5e-6, **kw)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            propagate_vector_hfpi_freespace_aperture(
                E, np.zeros_like(E), 5e-6, on_undersampled='silent', **kw)
        assert not [r for r in rec if 'UNDER-SAMPLED' in str(r.message)]

    @pytest.mark.parametrize('n_paths,strata,pre_alloc', [
        (100, (16, 16), 65536), (100, (32, 32), 1048576),
    ])
    def test_n_paths_is_a_cap_for_an_explicit_stratification(
            self, n_paths, strata, pre_alloc):
        """P2: ``n_per = max(1, n_paths // n_total)`` clamped to 1 and
        ``n_paths_actual = n_per * n_total >= n_total`` regardless of
        ``n_paths``, so an explicit stratification allocated up to
        10 486x the requested paths (76.6 MB from an n_paths=100 call).
        The docstring described the sub-sampling; the code never did it.
        """
        E = np.ones((16, 16), complex)
        b = init_paths_stratified(E, 2e-6, n_paths=n_paths, wavelength=LAM,
                                  rng=3, n_strata_xy=strata,
                                  n_strata_dir=strata)
        assert len(b) <= n_paths, (
            f'requested {n_paths} paths, allocated {len(b)} '
            f'(pre-fix {pre_alloc}).')

    def test_the_default_stratification_is_unaffected(self):
        """Counter-pin: the 4th-root rule already kept ``n_total`` near
        ``n_paths``, so the default path must still deliver about what
        was asked for."""
        E = np.ones((16, 16), complex)
        b = init_paths_stratified(E, 2e-6, n_paths=20000, wavelength=LAM,
                                  rng=3)
        assert 0.9 * 20000 <= len(b) <= 20000


# ===========================================================================
# K17 -- vectorial HFPI actually carries vector physics
# ===========================================================================

class TestK17VectorialHfpi:

    def test_the_rotation_conserves_energy_and_is_transverse(self):
        """P1: the v5.4.6 opt-in used the orthogonal projection and then
        DISCARDED the longitudinal component it created -- measured 8.8 %
        of the incident ``|E|^2`` lost at a 0.8 rad cone.  The rigid
        rotation is orthogonal, so both bars below are round-off, not
        physics: measured 1.1e-15 (norm) and 4.6e-16 (transversality),
        against a bar of 1e-12 -- three decades of margin, and ~11
        decades below the 8.8 % it replaces."""
        rng = np.random.default_rng(0)
        n = 5000
        th = rng.uniform(0.0, 0.8, n)
        ph = rng.uniform(0, 2 * np.pi, n)
        s_to = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                         np.cos(th)], axis=-1)
        s_from = np.stack([np.zeros(n), np.zeros(n), np.ones(n)], axis=-1)
        Ex = rng.normal(size=n) + 1j * rng.normal(size=n)
        Ey = rng.normal(size=n) + 1j * rng.normal(size=n)
        Ez = np.zeros(n, dtype=complex)
        rx, ry, rz = _rigid_rotate(Ex, Ey, Ez, s_from, s_to, np)
        p0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
        p1 = np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2
        assert float(np.max(np.abs(p1 / p0 - 1.0))) < 1e-12
        dot = np.abs(rx * s_to[:, 0] + ry * s_to[:, 1] + rz * s_to[:, 2])
        assert float(np.max(dot)) < 1e-12

    def test_the_rotation_reduces_to_the_richards_wolf_matrix(self):
        """Cross-check against a formula the module does not contain: for
        ``s_from = +z`` the rigid rotation must equal the aplanatic
        ``R_z(phi) R_y(theta) R_z(-phi)``.  Bar 1e-13 (round-off)."""
        th, ph = 0.6, 1.1
        s_to = np.array([[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                          np.cos(th)]])
        s_from = np.array([[0.0, 0.0, 1.0]])
        Ex = np.array([1.0 + 0j])
        Ey = np.array([0.0 + 0j])
        Ez = np.array([0.0 + 0j])
        rx, ry, rz = _rigid_rotate(Ex, Ey, Ez, s_from, s_to, np)
        c, s = np.cos(th), np.sin(th)
        cp, sp = np.cos(ph), np.sin(ph)
        assert complex(rx[0]) == pytest.approx(
            c * cp ** 2 + sp ** 2, abs=1e-13)
        assert complex(ry[0]) == pytest.approx(cp * sp * (c - 1), abs=1e-13)
        assert complex(rz[0]) == pytest.approx(-s * cp, abs=1e-13)

    def test_an_x_polarised_input_depolarises_and_grows_ez(self):
        """P1: the default path was bit-identical to two scalar HFPI runs
        -- max|Ex_vector - Ex_scalar| = 1.9e-23, zero cross-polarisation
        and no ``E_z`` anywhere.  Both are DECISIONS here (a component is
        identically zero, or it is not); the bars only have to sit above
        round-off, and the measured fractions are 1.4e-2 (|Ez|^2) and
        9.3e-5 (|Ey|^2) -- 13 and 11 decades above it."""
        N, dx = 24, 2e-6
        Ex_in = np.ones((N, N), dtype=complex)
        Ey_in = np.zeros((N, N), dtype=complex)
        kw = dict(z_to_aperture=200e-6, aperture_radius=40e-6,
                  z_aperture_to_output=200e-6, wavelength=LAM,
                  n_paths=200000, rng=17, cone_half_angle=0.35,
                  on_undersampled='silent')
        ex, ey, ez = propagate_vector_hfpi_freespace_aperture(
            Ex_in, Ey_in, dx, return_ez=True, **kw)
        I = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
        m = I > 0.02 * I.max()
        f_z = float(np.sum(np.abs(ez[m]) ** 2) / np.sum(I[m]))
        f_y = float(np.sum(np.abs(ey[m]) ** 2) / np.sum(I[m]))
        assert f_z > 1e-4, (
            f'|Ez|^2 fraction {f_z:.3e}: the module still carries no '
            f'longitudinal field.')
        assert f_y > 1e-8, (
            f'cross-polarised |Ey|^2 fraction {f_y:.3e}: an x-polarised '
            f'input still shows zero depolarisation.')

    def test_a_45_degree_input_depolarises_off_axis(self):
        """Pre-fix ``|Ey/Ex - 1| <= 1.1e-16`` over the WHOLE grid.  The
        bar (1e-3) sits 13 decades above that and 2 decades below the
        measured 0.143."""
        N, dx = 24, 2e-6
        a = 1 / np.sqrt(2)
        Ex_in = np.full((N, N), a, dtype=complex)
        Ey_in = np.full((N, N), a, dtype=complex)
        ex, ey = propagate_vector_hfpi_freespace_aperture(
            Ex_in, Ey_in, dx, z_to_aperture=200e-6, aperture_radius=40e-6,
            z_aperture_to_output=200e-6, wavelength=LAM, n_paths=200000,
            rng=17, cone_half_angle=0.35, on_undersampled='silent')
        I = np.abs(ex) ** 2 + np.abs(ey) ** 2
        m = I > 0.02 * I.max()
        spread = float(np.max(np.abs(ey[m] / ex[m] - 1.0)))
        assert spread > 1e-3, (
            f'max|Ey/Ex - 1| = {spread:.3e} over the grid: a 45-degree '
            f'input still shows no depolarisation anywhere.')

    def test_vector_projection_false_still_reproduces_two_scalar_runs(self):
        """Counter-pin: the legacy path stays available and stays what it
        was -- two scalar HFPI channels.  Bar: relative agreement 1e-12
        against the scalar propagator on the same seed (measured 1.1e-15
        absolute on an |Ex|max of 3.3)."""
        N, dx = 24, 2e-6
        Ex_in = np.ones((N, N), dtype=complex)
        kw = dict(z_to_aperture=200e-6, aperture_radius=40e-6,
                  z_aperture_to_output=200e-6, wavelength=LAM,
                  n_paths=100000, rng=17, cone_half_angle=0.35,
                  on_undersampled='silent')
        lex, _ley = propagate_vector_hfpi_freespace_aperture(
            Ex_in, np.zeros_like(Ex_in), dx, vector_projection=False, **kw)
        sx = propagate_hfpi_freespace_aperture(Ex_in, dx, **kw)
        rel = float(np.max(np.abs(lex - sx)) / np.max(np.abs(sx)))
        assert rel < 1e-12, rel


# ===========================================================================
# K14 / K24 -- MHS surface validation
# ===========================================================================

class TestK14MhsSurfaceCentre:

    def test_a_transverse_centre_jump_is_refused(self):
        """P3: ``_validate`` compared z / Ny / Nx / dx but not ``centre``,
        so a 50 um (and a 1 mm) transverse jump between subdomains was
        accepted and silently discarded -- while ``HuygensSurface.grid()``
        and ``aperture_subdomain`` both honour ``centre``."""
        s0 = HuygensSurface(z=0.0, Ny=16, Nx=16, dx=5e-6)
        s1 = HuygensSurface(z=1e-3, Ny=16, Nx=16, dx=5e-6)
        s1b = HuygensSurface(z=1e-3, Ny=16, Nx=16, dx=5e-6,
                             centre=(50e-6, 0.0))
        s2 = HuygensSurface(z=2e-3, Ny=16, Nx=16, dx=5e-6)
        MhsPipeline([asm_subdomain(s0, s1, wavelength=LAM),
                     asm_subdomain(s1, s2, wavelength=LAM)])
        with pytest.raises(ValueError, match='centre'):
            MhsPipeline([asm_subdomain(s0, s1, wavelength=LAM),
                         asm_subdomain(s1b, s2, wavelength=LAM)])
