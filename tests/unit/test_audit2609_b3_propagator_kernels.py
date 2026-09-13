"""WP-B3 regression pins -- AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.

The four propagator-kernel designs WP-A5 deferred (report section 6):

* **K13** -- ``propagate_hfpi_through_prescription`` gets an explicit
  output plane (``z_output``), which is what makes a photometric
  normalisation definable there at all.
* **K22** -- a Sobol sampler beside the jittered stratification, with the
  convergence of BOTH measured rather than assumed.
* **K9** (second half) -- the Shen-Wang pixel-integrated RS kernel,
  ``kernel='spatial-integrated'``.
* **K6** (second half) -- the band-limited chirp-Z resampler,
  ``resample_field(method='chirpz')``.

Every numeric bar below carries its oracle, that oracle's own error
floor, the measured values on 2026-09-13 and the decades of gap on each
side, per ``docs/TESTING_STANDARDS.md``.  No bar reads a clock: the
convergence claims are stated as error RATIOS at path counts or grid
counts that differ by a fixed factor.

Oracles used here are INDEPENDENT of the code under test:

* a super-sampled continuum RS-I double quadrature over the lit pixels
  of a piecewise-constant aperture (midpoint, no FFT, no padding, no
  library call) -- for the pixel-integrated kernel;
* the exact on-axis closed form ``U = e^{ikz} - (z/r_a) e^{ik r_a}``
  behind a circular aperture -- for the convergence order;
* an exact Hankel angular-spectrum quadrature for a Gaussian -- for the
  smooth-input ranking of the two spatial kernels;
* the Dirichlet-kernel (trigonometric) interpolant written out as an
  explicit double sum -- for the chirp-Z resampler;
* ``scipy.ndimage.map_coordinates`` driven directly -- for the byte
  identity of the resampler's default leg;
* band-limited ASM -- for the HFPI walk's photometric scale.

Two claims here need no oracle at all and are the stronger for it: an
unobstructed aperture plane must be transparent (so the two-leg walk
must equal the one-leg walk), and the closed walk must equal the public
``init_paths_stratified`` -> ``propagate_to_plane`` ->
``accumulate_to_grid`` sequence it is meant to be.

Author: Andrew Traverso -- WP-B3.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.ndimage import map_coordinates
from scipy.special import j0

from lumenairy.propagators.asm import angular_spectrum_propagate
from lumenairy.propagators.hfpi import (
    accumulate_to_grid,
    init_paths_stratified,
    propagate_hfpi_through_prescription,
    propagate_to_plane,
)
from lumenairy.propagators.mft import resample_field
from lumenairy.propagators.rs import (
    _RS_PIXEL_QUAD_NODES,
    _rs_pixel_integrated_kernel,
    rayleigh_sommerfeld_propagate,
    rs_alias_free_distance,
)

LAM = 633e-9


# ===========================================================================
# Shared oracles
# ===========================================================================

def _h_rs(X, Y, z, k):
    """The RS-I impulse response, written out here so every oracle below
    is independent of the module's own kernel build."""
    r = np.sqrt(X * X + Y * Y + z * z)
    return (z / (2 * np.pi * r ** 2)) * np.exp(1j * k * r) * (1.0 / r - 1j * k)


def _staircase_disc(N, dx, a):
    """Circular aperture as a pixel-centre indicator (the staircase)."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return (X * X + Y * Y <= a * a).astype(np.complex128)


def _grey_disc(N, dx, a, S=64):
    """The same aperture as its pixel-AREA average.

    Only the boundary band is sub-sampled: a pixel whose farthest corner
    is inside the circle is exactly 1 and one whose nearest corner is
    outside is exactly 0, so the ``S x S`` rule runs on ``O(N)`` pixels
    instead of ``O(N**2)``.  Its residual is the area of an edge pixel
    divided by ``S**2``, i.e. below ``1/S**2`` of one pixel -- 2.4e-4 at
    S = 64, three decades under the smallest field error it is used to
    measure (2.07e-5 relative on a field of order 2).
    """
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    h = 0.5 * dx
    d = np.hypot(np.abs(X) , np.abs(Y))
    r_far = np.hypot(np.abs(X) + h, np.abs(Y) + h)
    r_near = np.hypot(np.maximum(np.abs(X) - h, 0.0),
                      np.maximum(np.abs(Y) - h, 0.0))
    g = (r_far <= a).astype(np.float64)
    band = (r_near < a) & (r_far > a)
    del d
    if band.any():
        u = (np.arange(S) + 0.5) / S - 0.5
        xb, yb = X[band], Y[band]
        acc = np.zeros(xb.shape, dtype=np.float64)
        for i in range(S):
            yy = yb + u[i] * dx
            for j in range(S):
                xx = xb + u[j] * dx
                acc += (xx * xx + yy * yy <= a * a)
        g[band] = acc / (S * S)
    return g.astype(np.complex128)


def _rs_staircase_quadrature(E, dx, z, k, out_xy, S):
    """EXACT continuum RS-I integral of the piecewise-constant field ``E``.

    ``E`` is constant on each pixel ``[x_j +- dx/2]``, so the integral is
    a sum over the LIT pixels of ``Int_pixel h``.  Each pixel integral is
    an ``S x S`` super-sampled MIDPOINT rule -- deliberately a different
    quadrature from the module's Gauss-Legendre one, evaluated by a
    direct double sum at a handful of output points with no FFT, no
    padding and no library call.

    Its own floor is second order in ``1/S`` and is measured in the tests
    by running it at ``S`` and ``2S``.
    """
    N = E.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    lit = np.abs(E) > 0
    xs, ys, es = X[lit], Y[lit], E[lit]
    u = (np.arange(S) + 0.5) / S - 0.5
    du = dx / S
    out = []
    for (px, py) in out_xy:
        tot = 0.0 + 0.0j
        for a in range(S):
            dyy = py - (ys + u[a] * dx)
            for b in range(S):
                tot += np.sum(es * _h_rs(px - (xs + u[b] * dx), dyy, z, k))
        out.append(tot * du * du)
    return np.array(out)


def _hankel_gaussian(R, z, w0, lam=LAM, nf=4000):
    """EXACT propagated field of ``exp(-(r/w0)**2)`` at distance ``z``.

    Circularly symmetric, so the angular-spectrum integral collapses to a
    Hankel pair evaluated by Gauss-Legendre over ``f``.  No FFT, no grid,
    no library call.  Floor ~1e-12 relative (the Gaussian spectrum is
    1.6e-28 at the ``f_max = 8/(pi w0)`` cut and 4000 nodes resolve the
    ``J0`` oscillation to ~1e-12) -- four decades below the tightest bar
    asserted against it here.
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
    X, Y = np.meshgrid(x, x, indexing='xy')
    R = np.hypot(X, Y)
    return np.exp(-(R / w0) ** 2).astype(np.complex128), R, X, Y


def _plane_prescription(semi_diameter=np.inf, object_distance=0.0,
                        glass='air'):
    """One flat, index-matched surface in ``glass``: a plane the walk
    travels through, with an optional stop."""
    return {
        'object_distance': float(object_distance),
        'surfaces': [{'radius': np.inf, 'glass_before': glass,
                      'glass_after': glass,
                      'semi_diameter': semi_diameter}],
        'thicknesses': [0.0],
    }


def _ls_scale(out, ref, mask):
    """UNBIASED least-squares complex scale of ``out`` against ``ref``.

    The mean of ``|out|/|ref|`` is biased high for a Monte-Carlo estimate
    because ``|sum|**2`` carries the estimator's noise power; this
    projection does not.
    """
    den = float(np.sum(np.abs(ref[mask]) ** 2))
    return abs(complex(np.sum(np.asarray(out)[mask] * np.conj(ref[mask])))
               / den)


# ===========================================================================
# K13 -- the prescription walk's output plane
# ===========================================================================

class TestK13PrescriptionWalkOutputPlane:
    """WP-A5 deferred item 1.  The walk binned the bundle wherever the
    surface list left it, so a path re-emitted at a diffracting last
    surface had a zero-length final leg and the Huygens-Fresnel binning
    Jacobian ``r/(dx_out^2 cos theta_out)`` -- the K13 correction the
    free-space entry points apply -- had no ``r`` to use.  The walk
    therefore could not return a photometric amplitude at all, and said
    so in a warning.  ``z_output`` gives it the leg.
    """

    N, DX, W0, Z = 32, 4e-6, 12e-6, 2e-3

    def _fixture(self):
        E0, _R, _X, _Y = _gauss_grid(self.N, self.DX, self.W0)
        ref = angular_spectrum_propagate(E0, z=self.Z, wavelength=LAM,
                                         dx=self.DX)
        return E0, ref

    @pytest.mark.parametrize('pitch_mul,bar', [(1, 0.10), (2, 0.12)])
    def test_the_closed_walk_reproduces_asm(self, pitch_mul, bar):
        """Oracle: band-limited ASM on the same geometry, which is exact
        there (6.1e-8 relative L2 against the Hankel quadrature on this
        grid class).

        The walk is a single emission at ``z = 0`` through a flat
        index-matched plane, closed by ``z_output`` -- so ALL of the
        propagation is the new closing hop.

        Bar ``|scale - 1| < 0.10`` (``0.12`` at the coarser pitch).
        Derivation: six seeds at 2 M paths measured 0.9851 .. 0.9970
        (max deviation 0.0149) at ``dx_out = dx`` and 0.9559 .. 0.9689
        (0.0441) at ``2 dx``, so the bar sits 6.7x / 2.7x above the
        measured seed envelope.  The value this replaces -- the same
        walk with no output plane -- reads 1.16e-7, SEVEN decades below
        the bar on the other side.
        """
        E0, ref_fine = self._fixture()
        ref = ref_fine[::pitch_mul, ::pitch_mul]
        mask = np.abs(ref) > 0.05 * np.abs(ref).max()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = propagate_hfpi_through_prescription(
                E0, self.DX, _plane_prescription(), wavelength=LAM,
                n_paths=2_000_000, rng=1, cone_half_angle=0.05,
                output_dx=self.DX * pitch_mul,
                output_shape=(self.N // pitch_mul, self.N // pitch_mul),
                z_output=self.Z, on_undersampled='silent')
        scale = _ls_scale(out, ref, mask)
        assert abs(scale - 1.0) < bar, (
            f'closed walk / ASM least-squares scale = {scale:.4f} at '
            f'dx_out = {pitch_mul} dx; the walk with an output plane is '
            f'meant to BE the Huygens-Fresnel integral there.')

    def test_the_scale_does_not_move_with_path_count(self):
        """The K13 property itself: a corrected estimator's amplitude is
        a property of the physics, not of how many samples were spent.

        Bar: the 0.5 M and 8 M scales agree to 10 %.  Measured 0.9841 and
        0.9899 (seed-averaged over three seeds; 0.6 %), so the bar is
        ~17x the measured spread.  Pre-K13 the same quantity moved 14x
        between 2 M and 8 M paths -- two decades outside it.
        """
        E0, ref = self._fixture()
        mask = np.abs(ref) > 0.05 * np.abs(ref).max()
        scales = {}
        for n_paths in (500_000, 8_000_000):
            vals = []
            for seed in (1, 2, 3):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    out = propagate_hfpi_through_prescription(
                        E0, self.DX, _plane_prescription(), wavelength=LAM,
                        n_paths=n_paths, rng=seed, cone_half_angle=0.05,
                        output_dx=self.DX, output_shape=(self.N, self.N),
                        z_output=self.Z, on_undersampled='silent')
                vals.append(_ls_scale(out, ref, mask))
            scales[n_paths] = float(np.mean(vals))
        lo, hi = scales[500_000], scales[8_000_000]
        assert abs(hi / lo - 1.0) < 0.10, (
            f'scale moved from {lo:.4f} at 0.5 M paths to {hi:.4f} at '
            f'8 M -- a converged estimator does not do that.')

    def test_without_an_output_plane_the_walk_cannot_be_photometric(self):
        """FAIL-BEFORE.  The same call with no ``z_output`` bins the
        bundle at the plane the surfaces leave it on -- here the source
        plane -- so it is not the field at ``z`` by seven decades, and
        it warns.  Forcing ``'physical'`` there raises rather than
        returning zeros.
        """
        E0, ref = self._fixture()
        mask = np.abs(ref) > 0.05 * np.abs(ref).max()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = propagate_hfpi_through_prescription(
                E0, self.DX, _plane_prescription(), wavelength=LAM,
                n_paths=200_000, rng=1, cone_half_angle=0.05,
                output_dx=self.DX, output_shape=(self.N, self.N),
                on_undersampled='silent')
        assert any('NOT photometric' in str(w.message) for w in caught), (
            'the legacy path must still say that its amplitudes are not '
            'photometric.')
        scale = _ls_scale(out, ref, mask)
        assert scale < 1e-4, (
            f'un-propagated bundle / ASM scale = {scale:.3e}; measured '
            f'1.16e-7, and the closed walk reads ~0.99.')
        with pytest.raises(ValueError, match="normalisation='physical'"):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                propagate_hfpi_through_prescription(
                    E0, self.DX, _plane_prescription(), wavelength=LAM,
                    n_paths=20_000, rng=1, cone_half_angle=0.05,
                    output_dx=self.DX, output_shape=(self.N, self.N),
                    on_undersampled='silent', normalisation='physical')

    def test_an_open_stop_is_transparent(self):
        """Oracle-free property: a stop wider than the beam changes
        nothing, so the two-leg walk (emit, re-emit at the open stop,
        close on the output plane) must equal the one-leg walk over the
        same total distance.  This is the composition of
        ``_reemission_measure`` with ``_binning_jacobian``, which only
        the closing hop makes evaluable here.

        Bar: the two scales agree to 25 %.  Measured over four seeds at
        2 M paths and a 0.05 rad cone: two-leg 1.0428 +- 0.0499,
        one-leg 0.9908 +- 0.0041, ratio 1.0524 -- the bar is ~4.8x the
        measured deviation.  (The cascaded estimator's variance grows
        fast with the cone: the same ratio reads 2.29 at 0.12 rad and
        15.3 at 0.25 rad, which is why the bar is stated at a stated
        cone.)
        """
        E0, ref = self._fixture()
        mask = np.abs(ref) > 0.05 * np.abs(ref).max()
        two, one = [], []
        for seed in (1, 2, 3, 4):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                two.append(_ls_scale(propagate_hfpi_through_prescription(
                    E0, self.DX,
                    _plane_prescription(semi_diameter=10e-3,
                                        object_distance=self.Z / 2),
                    wavelength=LAM, n_paths=2_000_000, rng=seed,
                    cone_half_angle=0.05, output_dx=self.DX,
                    output_shape=(self.N, self.N), z_output=self.Z / 2,
                    on_undersampled='silent'), ref, mask))
                one.append(_ls_scale(propagate_hfpi_through_prescription(
                    E0, self.DX, _plane_prescription(), wavelength=LAM,
                    n_paths=2_000_000, rng=seed, cone_half_angle=0.05,
                    output_dx=self.DX, output_shape=(self.N, self.N),
                    z_output=self.Z, on_undersampled='silent'), ref, mask))
        ratio = float(np.mean(two)) / float(np.mean(one))
        assert abs(ratio - 1.0) < 0.25, (
            f'two-leg / one-leg = {ratio:.4f} through an OPEN stop '
            f'(means {np.mean(two):.4f} and {np.mean(one):.4f}); an '
            f'unobstructed aperture plane must be transparent.')

    def test_the_closing_hop_is_exactly_propagate_to_plane(self):
        """Structural pin, no physics oracle: the closed walk through a
        flat index-matched plane must equal the public three-call
        sequence ``init_paths_stratified`` ->
        ``propagate_to_plane(n_medium=n)`` -> ``accumulate_to_grid``,
        driven from the same spawned RNG stream.

        Run in GLASS, so the assertion also pins that the closing leg
        reads the prescription's ``glass_after`` index rather than
        assuming air: with ``n = 1`` the phases would differ by
        ``k (n - 1) z`` = 1.02e4 rad here, which is not a tolerance
        question.

        Bar: relative L2 below 1e-12.  The two routes do the same
        float64 arithmetic in the same order apart from the ray tracer's
        plane intersection at zero distance, so the residual is
        round-off; measured at the level of the bar's derivation this is
        an identity, not a comparison of two models.
        """
        from lumenairy.glass import get_glass_index
        from lumenairy.propagators.hfpi import _spawn_rng

        n_glass = float(get_glass_index('N-BK7', LAM))
        assert n_glass > 1.4
        E0, _R, _X, _Y = _gauss_grid(self.N, self.DX, self.W0)
        rx = _plane_prescription(glass='N-BK7')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            walk = propagate_hfpi_through_prescription(
                E0, self.DX, rx, wavelength=LAM, n_paths=200_000, rng=17,
                cone_half_angle=0.05, output_dx=self.DX,
                output_shape=(self.N, self.N), z_output=self.Z,
                on_undersampled='silent')
            paths = init_paths_stratified(
                E0, self.DX, n_paths=200_000, wavelength=LAM,
                rng=_spawn_rng(17, 0), cone_half_angle=0.05,
                z_input_plane=0.0)
            paths = propagate_to_plane(paths, z_target=self.Z,
                                       wavelength=LAM, n_medium=n_glass)
            hand = accumulate_to_grid(paths, Ny=self.N, Nx=self.N,
                                      dx=self.DX, on_undersampled='silent',
                                      normalisation='physical')
        rel = float(np.linalg.norm(np.asarray(walk) - np.asarray(hand))
                    / np.linalg.norm(np.asarray(hand)))
        assert rel < 1e-12, (
            f'closed walk vs the hand-built equivalent: relative L2 '
            f'{rel:.3e}.  A mismatch at the 1e-1 level would mean the '
            f'closing hop took the leg in air instead of N-BK7.')

    def test_a_powered_prescription_does_not_get_a_photometric_default(self):
        """The Jacobian both halves of the estimator apply is the
        FREE-SPACE ray-tube relation.  Through an element with power the
        system's own applies instead, and the measured amplitude error
        is 4879x at a 19.41 mm singlet's image plane -- so ``'auto'``
        must NOT resolve to ``'physical'`` there, and forcing it must
        warn.
        """
        import lumenairy as lm

        rx = lm.make_singlet(R1=20e-3, R2=-20e-3, d=1e-5, glass='N-BK7',
                             aperture=1e-3)
        rx['object_distance'] = 0.06
        E0, _R, _X, _Y = _gauss_grid(64, 8e-6, 60e-6)
        kw = dict(wavelength=LAM, n_paths=50_000, rng=5,
                  diffracting_surfaces=[], cone_half_angle=0.02,
                  output_dx=8e-6, output_shape=(64, 64), z_output=0.0287,
                  on_undersampled='silent')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            auto = propagate_hfpi_through_prescription(E0, 8e-6, rx, **kw)
        assert any('NOT photometric' in str(w.message) for w in caught), (
            "'auto' must fall back to the legacy sum -- and say so -- "
            "when the walk passes through an element with power.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            forced = propagate_hfpi_through_prescription(
                E0, 8e-6, rx, normalisation='physical', **kw)
        assert any('FREE-SPACE ray-tube Jacobian' in str(w.message)
                   for w in caught), (
            'forcing physical through a powered prescription must warn '
            'with what it costs.')
        # The two differ by the Jacobian, so this also proves 'auto' did
        # not quietly pick 'physical'.
        assert not np.allclose(np.asarray(auto), np.asarray(forced))

    def test_z_output_and_the_new_selectors_are_validated(self):
        E0, _R, _X, _Y = _gauss_grid(8, 4e-6, 12e-6)
        base = dict(wavelength=LAM, n_paths=64, rng=1,
                    cone_half_angle=0.05, output_dx=4e-6,
                    output_shape=(8, 8), on_undersampled='silent')
        for bad in (np.nan, np.inf):
            with pytest.raises(ValueError,
                               match='propagate_hfpi_through_prescription: '
                                     'z_output'):
                propagate_hfpi_through_prescription(
                    E0, 4e-6, _plane_prescription(), z_output=bad, **base)
        with pytest.raises(ValueError, match='sampler must be'):
            propagate_hfpi_through_prescription(
                E0, 4e-6, _plane_prescription(), sampler='nope', **base)
        with pytest.raises(ValueError, match='normalisation must be'):
            propagate_hfpi_through_prescription(
                E0, 4e-6, _plane_prescription(), normalisation='nope',
                **base)
        with pytest.raises(ValueError, match='sampling='):
            propagate_hfpi_through_prescription(
                E0, 4e-6, _plane_prescription(), sampler='sobol',
                sampling='uniform', **base)

    def test_both_new_keywords_reach_the_dispatcher(self):
        """``propagate(method='hfpi', prescription=…)`` forwards
        ``**kwargs`` verbatim, so both keywords are reachable from the
        library's front door.  Pinned because a keyword that only works
        on the sub-propagator is a keyword half the callers cannot use,
        and a future allow-list in the dispatcher would break it
        silently.
        """
        from lumenairy.propagators.dispatch import propagate

        E0, _R, _X, _Y = _gauss_grid(16, 4e-6, 12e-6)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = propagate(E0, dx=4e-6, wavelength=LAM, method='hfpi',
                            prescription=_plane_prescription(),
                            n_paths=4096, rng=1, cone_half_angle=0.05,
                            z_output=self.Z, sampler='sobol',
                            on_undersampled='silent')
        out = np.asarray(out)
        assert out.shape == (16, 16)
        assert np.any(np.abs(out) > 0)

    def test_a_reflective_last_surface_folds_the_closing_hop(self):
        """A mirror keeps the surrounding medium but reverses the
        direction, so the closing hop reaches a plane BEHIND it and kills
        every path sent to one in front.  Pins that the index lookup
        survives the ``'MIRROR'`` marker (which
        ``surfaces_from_prescription`` rewrites to ``glass_before``)
        rather than raising or silently reading air where there is glass.
        """
        E0, _R, _X, _Y = _gauss_grid(16, 4e-6, 12e-6)
        rx = _plane_prescription(object_distance=1e-3)
        rx['surfaces'][0]['glass_after'] = 'MIRROR'
        kw = dict(wavelength=LAM, n_paths=20_000, rng=1,
                  cone_half_angle=0.05, output_dx=4e-6,
                  output_shape=(16, 16), on_undersampled='silent')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            back = propagate_hfpi_through_prescription(
                E0, 4e-6, rx, z_output=-1e-3, **kw)
            fwd = propagate_hfpi_through_prescription(
                E0, 4e-6, rx, z_output=+1e-3, **kw)
        assert np.any(np.abs(np.asarray(back)) > 0), (
            'the reflected bundle must reach a plane behind the mirror')
        assert not np.any(np.abs(np.asarray(fwd)) > 0), (
            'no reflected path can reach a plane in front of the mirror')

    @pytest.mark.parametrize('rx_kind', ['stop_on_the_source_plane',
                                         'coincident_second_stop'])
    def test_a_zero_length_re_emission_does_not_get_a_photometric_default(
            self, rx_kind):
        """VERIFY-B3.  ``'auto'`` needs a THIRD answer, not two.
        ``_reemission_measure`` scales every re-emitted path by ``r_in``,
        the length of the leg that reached the surface; a stop sitting ON
        the plane the paths were last emitted from makes that zero for
        every path, and the measure refuses rather than returning zeros.
        Both prescriptions below are otherwise flat, index-matched and
        unsteered, so the first two conditions hold and ``'auto'`` used to
        resolve to ``'physical'`` and walk straight into that refusal.

        Property, no oracle: adding ``z_output`` to a call that worked
        must not turn it into an exception.  ``'auto'`` therefore falls
        back and says which condition failed, and the field it returns is
        BYTE-IDENTICAL to the same call with ``normalisation='legacy'``
        spelled out -- so the fallback is the legacy estimator itself, not
        a third thing.

        FAIL-BEFORE (both fixtures, measured on WP-B3 as committed):
        ``ValueError: apply_aperture_diffraction: normalisation='physical'
        needs the geometric length of the leg …`` -- raised from a private
        helper, naming a function the caller never called and offering
        ``init_paths_from_field`` as the remedy.  The second fixture is
        the ordinary one: ``object_distance`` is 1 mm, and it is the
        SECOND stop, at zero thickness behind the first, that has no leg.

        Forcing ``'physical'`` there still raises -- there is no factor to
        apply -- but now at the walk's own altitude, naming the surface.
        """
        E0, _R, _X, _Y = _gauss_grid(16, 4e-6, 12e-6)
        if rx_kind == 'stop_on_the_source_plane':
            rx = _plane_prescription(semi_diameter=50e-6,
                                     object_distance=0.0)
            bad_surface = 0
        else:
            rx = {'object_distance': 1e-3,
                  'surfaces': [{'radius': np.inf, 'glass_before': 'air',
                                'glass_after': 'air',
                                'semi_diameter': 60e-6},
                               {'radius': np.inf, 'glass_before': 'air',
                                'glass_after': 'air',
                                'semi_diameter': 60e-6}],
                  'thicknesses': [0.0, 0.0]}
            bad_surface = 1
        kw = dict(wavelength=LAM, n_paths=20_000, rng=1,
                  cone_half_angle=0.05, output_dx=4e-6,
                  output_shape=(16, 16), on_undersampled='silent')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            auto = propagate_hfpi_through_prescription(
                E0, 4e-6, rx, z_output=2e-3, **kw)
        msgs = [str(w.message) for w in caught]
        assert any('NOT photometric' in m for m in msgs), (
            "'auto' must fall back to the legacy sum when a re-emission "
            'has no incoming leg')
        assert any(f'surface {bad_surface} re-emits' in m for m in msgs), (
            f'the warning must name surface {bad_surface} as the one with '
            f'the zero-length leg; got {msgs!r}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            spelled = propagate_hfpi_through_prescription(
                E0, 4e-6, rx, z_output=2e-3, normalisation='legacy', **kw)
        assert np.asarray(auto).tobytes() == np.asarray(spelled).tobytes(), (
            "the 'auto' fallback must BE the legacy estimator")
        with pytest.raises(ValueError, match=(
                r"propagate_hfpi_through_prescription: "
                r"normalisation='physical'")):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                propagate_hfpi_through_prescription(
                    E0, 4e-6, rx, z_output=2e-3, normalisation='physical',
                    **kw)

    def test_the_legacy_warning_names_the_condition_that_failed(self):
        """VERIFY-B3.  Three independent conditions send the walk down the
        legacy branch, and a caller can hit more than one at once, so the
        diagnostic states the ones that applied instead of the commonest.

        Property, no oracle: a caller who HAS passed ``z_output`` must not
        be told to pass ``z_output``.

        FAIL-BEFORE (WP-B3 as committed): every legacy walk got the same
        sentence -- "This walk bins the bundle at the last surface rather
        than propagating it to a separate output plane … the last leg has
        zero length.  Pass z_output=…" -- including the singlet walk
        below, which passes ``z_output`` and does propagate to a separate
        plane, and the mirror walk, which does both and is folded.
        """
        import lumenairy as lm

        E0, _R, _X, _Y = _gauss_grid(16, 4e-6, 12e-6)
        kw = dict(wavelength=LAM, n_paths=20_000, rng=1,
                  cone_half_angle=0.05, output_dx=4e-6,
                  output_shape=(16, 16), on_undersampled='silent')

        def _why(rx, **extra):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                propagate_hfpi_through_prescription(E0, 4e-6, rx,
                                                    **kw, **extra)
            return '\n'.join(str(w.message) for w in caught
                             if 'NOT photometric' in str(w.message))

        no_plane = _why(_plane_prescription())
        assert 'no z_output was given' in no_plane
        assert 'element with power' not in no_plane

        rx_s = lm.make_singlet(R1=20e-3, R2=-20e-3, d=1e-5, glass='N-BK7',
                               aperture=1e-3)
        rx_s['object_distance'] = 0.06
        powered = _why(rx_s, z_output=0.0287, diffracting_surfaces=[])
        assert 'element with power' in powered
        assert 'no z_output was given' not in powered, (
            'this call passed z_output; the diagnostic must not ask for it')

        asked = _why(_plane_prescription(), normalisation='legacy')
        assert "normalisation='legacy' was asked for" in asked


# ===========================================================================
# K22 -- the Sobol sampler
# ===========================================================================

class TestK22SobolSampler:
    """WP-A5 deferred item 4.  ``sampler='sobol'`` places ``n_paths``
    scrambled Sobol points in the same 4-D
    ``(pixel_x, pixel_y, cos theta, phi)`` cube the jittered sampler
    stratifies.  The textbook QMC rate assumes a smooth integrand; this
    one has hard edges, so the rate is MEASURED here, not assumed.
    """

    N, DX, W0, Z = 32, 4e-6, 12e-6, 2e-3

    def _estimate(self, E0, n_paths, sampler, seed):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            paths = init_paths_stratified(
                E0, self.DX, n_paths=n_paths, wavelength=LAM, rng=seed,
                cone_half_angle=0.05, sampler=sampler)
            paths = propagate_to_plane(paths, z_target=self.Z,
                                       wavelength=LAM)
            return np.asarray(accumulate_to_grid(
                paths, Ny=self.N, Nx=self.N, dx=self.DX,
                on_undersampled='silent'))

    def _rms_err(self, E0, ref, mask, n_paths, sampler, seeds):
        a = np.array([self._estimate(E0, n_paths, sampler, s)[mask]
                      for s in seeds])
        return float(np.sqrt(np.mean(np.abs(a - ref[mask][None, :]) ** 2))
                     / np.sqrt(np.mean(np.abs(ref[mask]) ** 2)))

    @pytest.mark.parametrize('sampler', ['jittered', 'sobol'])
    def test_the_measured_rate_is_root_n_for_both_samplers(self, sampler):
        """The claim the changelog is allowed to make.

        Error ratio between path counts differing by 16x:
        ``O(N**-1/2)`` predicts 4.0 and ``O(N**-1)`` predicts 16.0.
        Measured over five INDEPENDENT three-seed groups:
        jittered 4.349 .. 4.533, sobol 4.628 .. 4.719.

        Bar ``3.2 < ratio < 8.0``: 1.36x below the smallest measurement,
        1.69x above the largest, and 2.0x below the ``O(N**-1)`` value
        the QMC advertisement would require.  No clock is read -- the
        independent variable is a path count.
        """
        E0, _R, _X, _Y = _gauss_grid(self.N, self.DX, self.W0)
        ref = angular_spectrum_propagate(E0, z=self.Z, wavelength=LAM,
                                         dx=self.DX)
        mask = np.abs(ref) > 0.05 * np.abs(ref).max()
        seeds = (7, 8, 9)
        e_lo = self._rms_err(E0, ref, mask, 1 << 15, sampler, seeds)
        e_hi = self._rms_err(E0, ref, mask, 1 << 19, sampler, seeds)
        ratio = e_lo / e_hi
        assert 3.2 < ratio < 8.0, (
            f'{sampler}: err(2**15)/err(2**19) = {ratio:.3f} '
            f'({e_lo:.4e} -> {e_hi:.4e}).  4.0 is Monte-Carlo, 16.0 is '
            f'the O(N**-1) a QMC claim would need.')

    def test_both_samplers_converge_to_the_same_field(self):
        """Oracle: band-limited ASM.  Whatever the rate, a scrambled
        Sobol sequence must be UNBIASED, so both samplers land on the
        same field.

        Bar ``|scale - 1| < 0.10`` for each.  Measured seed-mean scales
        over the convergence sweep: 0.9867 .. 1.0055 (jittered) and
        0.9899 .. 1.0061 (sobol) -- the bar is ~15x the measured
        deviation.
        """
        E0, _R, _X, _Y = _gauss_grid(self.N, self.DX, self.W0)
        ref = angular_spectrum_propagate(E0, z=self.Z, wavelength=LAM,
                                         dx=self.DX)
        mask = np.abs(ref) > 0.05 * np.abs(ref).max()
        got = {}
        for sampler in ('jittered', 'sobol'):
            vals = [_ls_scale(self._estimate(E0, 1 << 19, sampler, s),
                              ref, mask) for s in (7, 8, 9)]
            got[sampler] = float(np.mean(vals))
            assert abs(got[sampler] - 1.0) < 0.10, (
                f'{sampler} scale vs ASM = {got[sampler]:.4f}')
        assert abs(got['sobol'] / got['jittered'] - 1.0) < 0.10, (
            f'the two samplers disagree: {got}')

    def test_n_paths_is_exact_for_sobol_and_a_cap_for_jittered(self):
        E0 = np.ones((8, 8), dtype=complex)
        for n in (1000, 4096, 12345):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sob = init_paths_stratified(E0, 1e-6, n_paths=n,
                                            wavelength=LAM, rng=1,
                                            sampler='sobol')
                jit = init_paths_stratified(E0, 1e-6, n_paths=n,
                                            wavelength=LAM, rng=1)
            assert len(sob) == n, (
                f'sobol returned {len(sob)} paths for n_paths={n}')
            assert len(jit) <= n, (
                f'jittered returned {len(jit)} > {n}: n_paths is a cap.')

    def test_the_sobol_bundle_is_a_pure_function_of_rng(self):
        E0 = np.ones((8, 8), dtype=complex)
        kw = dict(n_paths=1024, wavelength=LAM, sampler='sobol')
        a = init_paths_stratified(E0, 1e-6, rng=3, **kw)
        b = init_paths_stratified(E0, 1e-6, rng=3, **kw)
        c = init_paths_stratified(E0, 1e-6, rng=4, **kw)
        assert a.positions.tobytes() == b.positions.tobytes()
        assert a.directions.tobytes() == b.directions.tobytes()
        assert a.positions.tobytes() != c.positions.tobytes(), (
            'a different seed must give a different scramble -- otherwise '
            'the "re-run and compare" error estimate returns zero.')

    def test_the_sobol_points_fill_the_cube(self):
        """A mis-mapped cube would pile the paths into a corner.  Every
        one of the four coordinates must span its own range and be
        uniform to the discrepancy of 4096 points.
        """
        Ny, Nx = 16, 12
        E0 = np.ones((Ny, Nx), dtype=complex)
        cone = 0.4
        p = init_paths_stratified(E0, 1e-6, n_paths=4096, wavelength=LAM,
                                  rng=2, cone_half_angle=cone,
                                  sampler='sobol')
        ix = np.round(np.asarray(p.positions)[:, 0] / 1e-6 + Nx / 2)
        iy = np.round(np.asarray(p.positions)[:, 1] / 1e-6 + Ny / 2)
        assert set(np.unique(ix)) == set(range(Nx))
        assert set(np.unique(iy)) == set(range(Ny))
        cos_t = np.asarray(p.directions)[:, 2]
        cos_max = np.cos(cone)
        # Uniform in cos(theta) over [cos_max, 1] -> mean at the midpoint.
        # 4096 Sobol points put the mean within 1e-3 of it; the bar is 30x
        # that, and a sampler that collapsed the axis would read 0 or 1.
        u = (cos_t - cos_max) / (1.0 - cos_max)
        assert abs(float(u.mean()) - 0.5) < 0.03, float(u.mean())
        assert u.min() < 0.01 and u.max() > 0.99
        phi = np.arctan2(np.asarray(p.directions)[:, 1],
                         np.asarray(p.directions)[:, 0])
        assert abs(float(np.mean(np.cos(phi)))) < 0.05
        assert abs(float(np.mean(np.sin(phi)))) < 0.05

    def test_the_sampler_selectors_are_validated(self):
        E0 = np.ones((8, 8), dtype=complex)
        with pytest.raises(ValueError,
                           match='init_paths_stratified: sampler must be'):
            init_paths_stratified(E0, 1e-6, n_paths=16, wavelength=LAM,
                                  sampler='halton')
        with pytest.raises(ValueError, match='n_strata_xy'):
            init_paths_stratified(E0, 1e-6, n_paths=16, wavelength=LAM,
                                  sampler='sobol', n_strata_xy=(2, 2))
        with pytest.warns(UserWarning, match='not a power of two'):
            init_paths_stratified(E0, 1e-6, n_paths=1000, wavelength=LAM,
                                  rng=1, sampler='sobol')


# ===========================================================================
# K9 (second half) -- the pixel-integrated Rayleigh-Sommerfeld kernel
# ===========================================================================

class TestK9PixelIntegratedRsKernel:
    """WP-A5 deferred item 2.  ``kernel='spatial-integrated'`` integrates
    the RS-I Green's function over each pixel instead of sampling it at
    the centre (Shen & Wang 2006).  It is the exact operator for an input
    whose staircase IS the object, and NOT a drop-in accuracy upgrade for
    a sampled smooth field -- which is why the default does not move.
    """

    #: Window, aperture radius and distance.  ``z = 16 mm`` is above the
    #: alias threshold ``2 N dx**2 / lambda`` of EVERY grid used below
    #: (12.94 mm at the coarsest, N = 64 / dx = 8 um), so both spatial
    #: kernels are legal on all of them, and it puts the Fresnel number
    #: ``a**2/(lambda z)`` at 0.99 -- the on-axis maximum, so the closed
    #: form it is compared against is order 2 rather than near a zero.
    L, A, Z = 512e-6, 100e-6, 16e-3

    def test_the_default_and_the_point_kernel_are_still_the_same_array(self):
        """The routing is untouched: above the alias threshold ``'auto'``
        IS ``'spatial'``, bit for bit, and the new token does not sit in
        its way."""
        for (N, dx, z) in ((64, 2e-6, 2e-3), (128, 1e-6, 3e-3),
                           (65, 1.5e-6, 9e-4)):
            E0, _R, _X, _Y = _gauss_grid(N, dx, 6e-6)
            assert z >= rs_alias_free_distance(N, dx, LAM)
            auto = rayleigh_sommerfeld_propagate(E0, z, LAM, dx)
            spatial = rayleigh_sommerfeld_propagate(E0, z, LAM, dx,
                                                    kernel='spatial')
            assert auto.tobytes() == spatial.tobytes()

    def test_the_integrated_kernel_is_the_exact_cell_constant_integral(self):
        """Oracle: the continuum RS-I double quadrature of the SAME
        staircase aperture, by a different rule (super-sampled midpoint),
        at four output points, with no FFT and no library call.

        The oracle's own floor is measured here by running it at ``S``
        and ``2S``: it is second order in ``1/S``, so the difference IS
        the floor to within a factor 4/3.

        Two-sided.  Measured (N = 64, dx = 8 um, a = 100 um, z = 16 mm):
        oracle floor 1.3794e-5 at S = 16 and 3.4485e-6 at S = 32; the
        integrated kernel reads 4.5979e-6 / 1.1495e-6 -- BELOW the floor
        and dividing by four every time S doubles, which is the oracle
        converging onto it rather than the kernel moving -- while the
        point-sampled kernel sits at 4.7464e-3 / 4.7499e-3, unmoved,
        1032x / 4132x above.
        """
        N, dx = 64, self.L / 64
        assert self.Z >= rs_alias_free_distance(N, dx, LAM)
        k = 2 * np.pi / LAM
        E = _staircase_disc(N, dx, self.A)
        pts_px = [(0, 0), (3, 0), (7, 4), (13, 0)]
        idx = [(N // 2 + j, N // 2 + i) for (i, j) in pts_px]
        xy = [((c - N / 2) * dx, (r - N / 2) * dx) for (r, c) in idx]
        o_lo = _rs_staircase_quadrature(E, dx, self.Z, k, xy, 16)
        o_hi = _rs_staircase_quadrature(E, dx, self.Z, k, xy, 32)
        floor = float(np.linalg.norm(o_lo - o_hi) / np.linalg.norm(o_hi))
        assert floor < 1e-4, f'oracle floor {floor:.3e} is too loose to pin'

        got = {}
        for kern in ('spatial-integrated', 'spatial'):
            out = rayleigh_sommerfeld_propagate(E, self.Z, LAM, dx,
                                                kernel=kern)
            vals = np.array([out[r, c] for (r, c) in idx])
            got[kern] = float(np.linalg.norm(vals - o_hi)
                              / np.linalg.norm(o_hi))
        assert got['spatial-integrated'] < 2.0 * floor, (
            f"integrated kernel {got['spatial-integrated']:.3e} against an "
            f'oracle whose own floor is {floor:.3e} (measured 4.598e-6 '
            f'against 1.379e-5, i.e. a third of it).')
        assert got['spatial'] > 100.0 * got['spatial-integrated'], (
            f"point kernel {got['spatial']:.3e} vs integrated "
            f"{got['spatial-integrated']:.3e}: measured 1032x, and the two "
            f'must be reading the same input differently, by decades.')

    def test_the_point_kernel_is_the_right_one_for_a_sampled_smooth_field(
            self):
        """The reason the default does NOT move.  Oracle: the exact
        Hankel angular-spectrum quadrature for a Gaussian (floor ~1e-12,
        four decades below the tighter of the two values).

        Measured on six legal geometries: ``'spatial'`` 3.2e-8 .. 6.6e-8
        against ``'spatial-integrated'`` 8.2e-4 .. 1.3e-2 -- the
        integrated kernel is four to five decades worse here, because it
        convolves the staircase of the Gaussian rather than the Gaussian.
        Bar: the point kernel is at least 100x closer, which is 2.3
        decades inside the smallest measured gap (8.18e-4 / 6.35e-8 =
        1.3e4, at N = 256 / dx = 0.5 um / z = 0.25 mm -- not run here
        because the Hankel oracle costs 20 s at that grid).
        """
        for (N, dx, w0, z) in ((128, 1e-6, 6e-6, 3e-3),
                               (64, 2e-6, 6e-6, 9e-4)):
            E0, R, _X, _Y = _gauss_grid(N, dx, w0)
            assert z >= rs_alias_free_distance(N, dx, LAM)
            ref = _hankel_gaussian(R, z, w0)
            nref = np.linalg.norm(ref)
            e_pt = float(np.linalg.norm(
                rayleigh_sommerfeld_propagate(E0, z, LAM, dx,
                                              kernel='spatial') - ref) / nref)
            e_in = float(np.linalg.norm(
                rayleigh_sommerfeld_propagate(
                    E0, z, LAM, dx, kernel='spatial-integrated') - ref)
                / nref)
            assert e_in > 100.0 * e_pt, (
                f'N={N} dx={dx}: point {e_pt:.3e}, integrated {e_in:.3e}.  '
                f'If this ever inverts, the default routing should be '
                f'revisited -- it is the reason auto keeps the point '
                f'kernel.')

    def test_what_restores_second_order_is_the_apertures_edge(self):
        """The audit's roughly-first-order convergence on a hard aperture
        is the aperture's EDGE, not the kernel.  Oracle: the exact
        on-axis closed form ``U = e^{ikz} - (z/r_a) e^{ik r_a}``, which
        is algebraic -- no floor of its own beyond float64.

        Measured orders between successive N (128, 256, 512, 1024) at
        z = 16 mm: pixel-centre indicator 1.307 / 3.294 / -0.624 (point
        kernel) and 1.321 / 3.258 / -0.582 (integrated) -- erratic and
        non-monotone, because a circle's staircase area error does not
        shrink smoothly; exact pixel-AREA average 2.035 / 2.024 / 2.027
        and 2.016 / 2.013 / 2.012, clean second order for BOTH kernels.

        Bar: each grey order in ``[1.75, 2.25]``.  The measured spread
        about 2 is 0.035, so the bar is 7x it; first order (1.0) and
        third (3.0) are both 3x outside.  The same assertion applied to
        the staircase arm would fail on its first pair, which is the
        point of the test.
        """
        k = 2 * np.pi / LAM
        r_a = np.sqrt(self.A ** 2 + self.Z ** 2)
        exact = np.exp(1j * k * self.Z) - (self.Z / r_a) * np.exp(1j * k * r_a)
        errs = {'spatial': [], 'spatial-integrated': []}
        stair = {'spatial': [], 'spatial-integrated': []}
        for N in (128, 256, 512):
            dx = self.L / N
            assert self.Z >= rs_alias_free_distance(N, dx, LAM)
            grey = _grey_disc(N, dx, self.A)
            hard = _staircase_disc(N, dx, self.A)
            for kern in errs:
                for src, dest in ((grey, errs), (hard, stair)):
                    out = rayleigh_sommerfeld_propagate(src, self.Z, LAM, dx,
                                                        kernel=kern)
                    dest[kern].append(
                        abs(complex(out[N // 2, N // 2]) - exact)
                        / abs(exact))
        for kern, e in errs.items():
            orders = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
            assert all(1.75 < o < 2.25 for o in orders), (
                f'{kern} with an area-averaged aperture: measured orders '
                f'{[f"{o:.2f}" for o in orders]} from errors '
                f'{[f"{v:.3e}" for v in e]}; second order is the claim.')
        # Falsifiability (TESTING_STANDARDS V1): the same assertion must
        # FAIL on the staircase arm, or the bar above is measuring
        # nothing.  Measured orders there are 1.307 and 3.294 (point) /
        # 1.321 and 3.258 (integrated) -- both outside the band by at
        # least 0.44.
        for kern, e in stair.items():
            orders = [np.log2(e[i] / e[i + 1]) for i in range(len(e) - 1)]
            assert not all(1.75 < o < 2.25 for o in orders), (
                f'{kern} with a pixel-centre-indicator aperture also read '
                f'second order ({[f"{o:.2f}" for o in orders]}); then the '
                f'aperture edge is NOT what limits this and the claim '
                f'above needs re-deriving.')

    def test_the_pixel_quadrature_is_converged_at_its_node_count(self):
        """The shipped node count must put the pixel integral's own error
        decades below anything it is used to measure.

        Bar: relative L2 against a 10-node build below 1e-9, at the WORST
        legal geometry ``z = 2 N dx^2 / lambda`` where the kernel's phase
        sweeps its full ``pi`` across a pixel.  Measured 2.7e-11 ..
        4.7e-11 at the shipped 6 nodes (and 1.5e-4 at 3 nodes, seven
        decades worse, which is what the bar is separating) -- those
        figures come from the 14-node reference tabulated at
        :data:`~lumenairy.propagators.rs._RS_PIXEL_QUAD_NODES`; 10 nodes
        is used here because both are decades past convergence and it
        costs less.
        """
        k = 2 * np.pi / LAM
        assert _RS_PIXEL_QUAD_NODES >= 4
        for (N, dx) in ((64, 2e-6), (128, 1e-6)):
            z = rs_alias_free_distance(N, dx, LAM)
            ref = _rs_pixel_integrated_kernel(2 * N, 2 * N, dx, dx, z, k, np,
                                              n_nodes=10)
            got = _rs_pixel_integrated_kernel(2 * N, 2 * N, dx, dx, z, k, np)
            rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
            assert rel < 1e-9, (
                f'N={N} dx={dx}: {_RS_PIXEL_QUAD_NODES}-node build differs '
                f'from a 10-node one by {rel:.3e}.')
            coarse = _rs_pixel_integrated_kernel(2 * N, 2 * N, dx, dx, z, k,
                                                 np, n_nodes=3)
            assert (float(np.linalg.norm(coarse - ref) / np.linalg.norm(ref))
                    > 100.0 * max(rel, 1e-15)), (
                'a 3-node build must be visibly worse, or this test is '
                'measuring nothing.')

    def test_the_folded_build_equals_an_unfolded_one(self):
        """The kernel is built on one quadrant and mirrored because ``h``
        depends on ``x`` and ``y`` only through ``x**2 + y**2``.  Oracle:
        the same Gauss-Legendre rule written out here over the FULL
        padded grid.

        Bar: relative L2 below 1e-14.  Measured 2.1e-16 on three grids --
        not bit-identical, because the mirrored node sum runs in the
        reverse order, so the residual is exactly float64 round-off and
        1e-14 is ~50x above it.
        """
        k = 2 * np.pi / LAM
        t, w = np.polynomial.legendre.leggauss(_RS_PIXEL_QUAD_NODES)
        for (N, dx, dy, z) in ((32, 2e-6, 2e-6, 9e-4),
                               (48, 1.5e-6, 3e-6, 1.2e-3)):
            n2y = n2x = 2 * N
            x = (np.arange(n2x) - n2x / 2) * dx
            y = (np.arange(n2y) - n2y / 2) * dy
            acc = np.zeros((n2y, n2x), dtype=complex)
            for a in range(len(t)):
                ya = y + 0.5 * dy * t[a]
                for b in range(len(t)):
                    X, Y = np.meshgrid(x + 0.5 * dx * t[b], ya, indexing='xy')
                    acc += (w[a] * w[b]) * _h_rs(X, Y, z, k)
            acc *= 0.25 * dx * dy
            got = _rs_pixel_integrated_kernel(n2y, n2x, dy, dx, z, k, np)
            rel = float(np.linalg.norm(got - acc) / np.linalg.norm(acc))
            assert rel < 1e-14, (
                f'folded vs unfolded pixel-integrated kernel: {rel:.3e} '
                f'at N={N}, dx={dx}, dy={dy}.')

    def test_the_two_spatial_kernels_do_not_share_a_cache_slot(self):
        """Both are cached on the padded geometry; a shared tag would
        hand the second caller the first one's array."""
        N, dx, z = 64, 2e-6, 2e-3
        E0, _R, _X, _Y = _gauss_grid(N, dx, 6e-6)
        first = rayleigh_sommerfeld_propagate(E0, z, LAM, dx,
                                              kernel='spatial')
        integ = rayleigh_sommerfeld_propagate(E0, z, LAM, dx,
                                              kernel='spatial-integrated')
        again = rayleigh_sommerfeld_propagate(E0, z, LAM, dx,
                                              kernel='spatial')
        integ2 = rayleigh_sommerfeld_propagate(E0, z, LAM, dx,
                                               kernel='spatial-integrated')
        assert first.tobytes() == again.tobytes()
        assert integ.tobytes() == integ2.tobytes()
        assert first.tobytes() != integ.tobytes()

    def test_the_alias_guard_and_the_token_cover_the_new_kernel(self):
        N, dx = 64, 2e-6
        z = 0.5 * rs_alias_free_distance(N, dx, LAM)
        E0, _R, _X, _Y = _gauss_grid(N, dx, 6e-6)
        with pytest.raises(ValueError, match='ALIASES'):
            rayleigh_sommerfeld_propagate(E0, z, LAM, dx,
                                          kernel='spatial-integrated')
        with pytest.raises(ValueError, match='ALIASES'):
            rayleigh_sommerfeld_propagate(E0, z, LAM, dx, kernel='spatial')
        with pytest.raises(ValueError,
                           match='rayleigh_sommerfeld_propagate: kernel'):
            rayleigh_sommerfeld_propagate(E0, 2e-3, LAM, dx,
                                          kernel='spatial_integrated')

    def test_the_new_kernel_reaches_the_hf_free_space_entry_point(self):
        """``propagate_huygens_fresnel_freespace`` is a delegation, so the
        keyword must arrive intact rather than being swallowed."""
        from lumenairy.propagators.hf import (
            propagate_huygens_fresnel_freespace,
        )
        N, dx, z = 64, 2e-6, 2e-3
        E0, _R, _X, _Y = _gauss_grid(N, dx, 6e-6)
        via_hf = propagate_huygens_fresnel_freespace(
            E0, z, LAM, dx, kernel='spatial-integrated')
        direct = rayleigh_sommerfeld_propagate(
            E0, z, LAM, dx, kernel='spatial-integrated')
        assert np.asarray(via_hf).tobytes() == direct.tobytes()


# ===========================================================================
# K6 (second half) -- the band-limited (chirp-Z) resampler
# ===========================================================================

class TestK6ChirpZResampler:
    """WP-A5 deferred item 3.  ``resample_field(method='chirpz')``
    evaluates the band-limited interpolant of the samples on the output
    grid instead of a cubic spline through them, so its MTF is exactly 1
    where the spline's rolls off to 0.72.
    """

    N, DX, W0 = 64, 1e-6, 12e-6

    def _carrier(self, cyc_per_px, N=None, dx=None):
        N = self.N if N is None else N
        dx = self.DX if dx is None else dx
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        env = np.exp(-(X ** 2 + Y ** 2) / self.W0 ** 2)
        return (env * np.exp(2j * np.pi * cyc_per_px * X / dx)).astype(complex)

    def test_the_default_leg_is_the_historical_spline(self):
        """Oracle: ``map_coordinates`` driven directly with the
        coordinate map the docstring states.  Bar: BYTE identity -- there
        is no tolerance to derive for "this keyword selects, it does not
        modify".
        """
        for (N, dx_in, dx_out, n_out, order) in ((64, 1e-6, 0.5e-6, 128, 3),
                                                 (64, 1e-6, 1.5e-6, 64, 3),
                                                 (65, 1e-6, 0.7e-6, 93, 1),
                                                 (64, 1e-6, 2e-6, 32, 5)):
            E = self._carrier(0.2, N=N, dx=dx_in)
            scale = dx_out / dx_in
            ix = (np.arange(n_out) - n_out / 2) * scale + N / 2
            IX, IY = np.meshgrid(ix, ix)
            coords = np.array([IY.ravel(), IX.ravel()])
            want = (map_coordinates(E.real, coords, order=order,
                                    mode='constant', cval=0.0)
                    + 1j * map_coordinates(E.imag, coords, order=order,
                                           mode='constant', cval=0.0)
                    ).reshape(n_out, n_out)
            got, dxo = resample_field(E, dx_in, dx_out, n_out, order)
            assert got.tobytes() == want.tobytes(), (
                f'default leg moved at N={N} order={order}')
            assert dxo == dx_out

    def test_the_chirpz_mtf_is_flat_where_the_splines_rolls_off(self):
        """Oracle: Parseval on a pure carrier -- an exact resampler
        returns the same power at every frequency the grid represents.

        Measured (Gaussian envelope, N = 64, dx = 1 um, w0 = 12 um):

        =========  ============  ==========  ==========
        cyc/px     px per cycle  spline      chirpz
        =========  ============  ==========  ==========
        0.10       10            0.999497    1.000000
        0.20       5             0.990258    1.000000
        0.30       3.33          0.930135    1.000000
        0.40       2.5           0.717718    1.000000
        =========  ============  ==========  ==========

        Bars: chirpz within 1e-6 of 1 at every carrier (measured
        deviation < 1e-9, so the bar is three decades up), and the spline
        below 0.80 at 0.40 cyc/px (measured 0.7177, and 1.0 is the value
        it would need to have no MTF at all).
        """
        for cyc in (0.0, 0.10, 0.20, 0.30, 0.40):
            E = self._carrier(cyc)
            p_in = float(np.sum(np.abs(E) ** 2)) * self.DX ** 2
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sp, _ = resample_field(E, self.DX, self.DX / 2, 2 * self.N)
                cz, _ = resample_field(E, self.DX, self.DX / 2, 2 * self.N,
                                       method='chirpz')
            r_cz = (float(np.sum(np.abs(cz) ** 2))
                    * (self.DX / 2) ** 2 / p_in)
            r_sp = (float(np.sum(np.abs(sp) ** 2))
                    * (self.DX / 2) ** 2 / p_in)
            assert abs(r_cz - 1.0) < 1e-6, (
                f'chirpz P_out/P_in = {r_cz:.8f} at {cyc} cyc/px')
            if cyc >= 0.40:
                assert r_sp < 0.80, (
                    f'spline P_out/P_in = {r_sp:.6f} at {cyc} cyc/px; the '
                    f'roll-off this option exists for has vanished.')

    def test_chirpz_is_the_trigonometric_interpolant(self):
        """Oracle: the Dirichlet-kernel interpolant written out as an
        explicit double sum over the centred DFT bins -- the definition,
        evaluated with no Bluestein, no chirp and no cache.

        Bar: relative L2 below 1e-12.  Both routes are float64 sums of
        the same ``N_in**2`` terms in different orders, so the residual
        is round-off; measured at this size it is ~1e-14.
        """
        N, dx_in, dx_out, n_out = 16, 1e-6, 0.6e-6, 21
        rng = np.random.default_rng(5)
        E = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
        A = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E)))
        f = (np.arange(N) - N // 2) / (N * dx_in)
        off = (N / 2.0 - N // 2) * dx_in
        xo = (np.arange(n_out) - n_out / 2) * dx_out + off
        phase_x = np.exp(2j * np.pi * np.outer(xo, f))
        want = (phase_x @ A @ phase_x.T) / (N * N)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got, _ = resample_field(E, dx_in, dx_out, n_out, method='chirpz')
        rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        assert rel < 1e-12, f'chirpz vs the explicit interpolant: {rel:.3e}'

    def test_chirpz_reproduces_the_samples_it_was_given(self):
        """``dx_out == dx_in`` is a forward-and-inverse transform pair, so
        the output must be the input to FFT round-off.

        Bar: relative L2 below 1e-12 (measured 9.5e-15; the spline leg's
        own identity is 2.3e-16, and both are decades under the bar).
        """
        E = self._carrier(0.3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got, _ = resample_field(E, self.DX, self.DX, self.N,
                                    method='chirpz')
        rel = float(np.linalg.norm(got - E) / np.linalg.norm(E))
        assert rel < 1e-12, f'chirpz identity relative L2 {rel:.3e}'

    def test_a_window_wider_than_the_period_returns_replicas_and_warns(self):
        """The chirp-Z reconstruction is periodic with period
        ``N_in*dx_in``; the spline leg pads with zeros instead.  A caller
        switching a call site has to know which of those it is getting.

        Bar: the warning fires, and the returned power is 4x (the 2x2
        replica tiling) to within 1e-6 -- an exact, derived number, not a
        tolerance.
        """
        E = self._carrier(0.0, N=128, dx=1e-6)
        p_in = float(np.sum(np.abs(E) ** 2)) * (1e-6) ** 2
        with pytest.warns(UserWarning, match='faithful zone'):
            wide, _ = resample_field(E, 1e-6, 2e-6, 128, method='chirpz')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            zero_pad, _ = resample_field(E, 1e-6, 2e-6, 128)
        assert not caught
        assert abs(float(np.sum(np.abs(wide) ** 2)) * (2e-6) ** 2 / p_in
                   - 4.0) < 1e-6
        assert abs(float(np.sum(np.abs(zero_pad) ** 2)) * (2e-6) ** 2 / p_in
                   - 1.0) < 1e-3

    def test_the_method_token_is_validated(self):
        E = self._carrier(0.1)
        with pytest.raises(ValueError,
                           match='resample_field: method must be'):
            resample_field(E, self.DX, self.DX / 2, 128, method='fft')

    def test_a_non_square_input_keeps_both_axes(self):
        """``resample_field`` takes ONE pitch but not one shape, and the
        extent-preserving default sizes each axis separately."""
        Ny, Nx = 48, 64
        x = (np.arange(Nx) - Nx / 2) * self.DX
        y = (np.arange(Ny) - Ny / 2) * self.DX
        X, Y = np.meshgrid(x, y, indexing='xy')
        E = np.exp(-(X ** 2 + Y ** 2) / self.W0 ** 2).astype(complex)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got, _ = resample_field(E, self.DX, self.DX / 2, None,
                                    method='chirpz')
        assert got.shape == (2 * Ny, 2 * Nx)
        p_in = float(np.sum(np.abs(E) ** 2)) * self.DX ** 2
        p_out = float(np.sum(np.abs(got) ** 2)) * (self.DX / 2) ** 2
        assert abs(p_out / p_in - 1.0) < 1e-9

    @pytest.mark.parametrize('Ny,Nx,dx_out_mul', [(65, 65, 0.5),
                                                  (45, 63, 0.7),
                                                  (33, 21, 1.4),
                                                  (17, 17, 0.55)])
    def test_the_chirpz_leg_places_an_odd_grids_origin(self, Ny, Nx,
                                                       dx_out_mul):
        """VERIFY-B3.  ``ifftshift`` puts the spectrum's spatial origin at
        the INTEGER index ``N//2`` while this family's declared grid is
        ``x = (n - N/2) d``, so for ODD ``N`` the two differ by half an
        input pixel and the leg has to carry that offset.  Both halves of
        the handling are pinned here: the ``off_in`` shift folded into the
        output centre, and the ``N_in // 2`` frequency-bin centre the
        ``fftshift`` actually produces.

        Oracle: the Dirichlet-kernel interpolant as an explicit double sum
        over the centred DFT bins, written out below for a NON-SQUARE grid
        so each axis is placed on its own count -- the definition, with no
        Bluestein, no chirp and no cache.

        Bar: relative L2 below 1e-12.  Both routes are float64 sums of the
        same ``Ny_in*Nx_in`` terms in a different order, so the residual is
        round-off; measured 6.0e-15 .. 1.8e-14 on these four grids.

        FAIL-BEFORE, measured on the same four fixtures: dropping the
        half-pixel offset gives 1.077 .. 1.098, and using ``N_in/2``
        instead of ``N_in//2`` for the input bin centre gives 1.038 ..
        1.094 -- a 100 %-class error, fourteen decades outside the bar, and
        exactly 0 on every EVEN grid, which is why an even-N pin cannot
        see either of them.
        """
        dx_in = self.DX
        dx_out = self.DX * dx_out_mul
        rng = np.random.default_rng(11)
        E = (rng.normal(size=(Ny, Nx)) + 1j * rng.normal(size=(Ny, Nx)))
        Nx_out = int(round(Nx * dx_in / dx_out))
        Ny_out = int(round(Ny * dx_in / dx_out))

        fx = (np.arange(Nx) - Nx // 2) / (Nx * dx_in)
        fy = (np.arange(Ny) - Ny // 2) / (Ny * dx_in)
        x_in = (np.arange(Nx) - Nx / 2.0) * dx_in
        y_in = (np.arange(Ny) - Ny / 2.0) * dx_in
        A = (np.exp(-2j * np.pi * np.outer(fy, y_in)) @ E
             @ np.exp(-2j * np.pi * np.outer(fx, x_in)).T)
        x_out = (np.arange(Nx_out) - Nx_out / 2.0) * dx_out
        y_out = (np.arange(Ny_out) - Ny_out / 2.0) * dx_out
        want = (np.exp(2j * np.pi * np.outer(y_out, fy)) @ A
                @ np.exp(2j * np.pi * np.outer(x_out, fx)).T) / (Nx * Ny)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            got, _ = resample_field(E, dx_in, dx_out, None, method='chirpz')
        assert got.shape == (Ny_out, Nx_out)
        rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        assert rel < 1e-12, (
            f'chirpz vs the explicit interpolant on a {Ny}x{Nx} grid at '
            f'dx_out/dx_in = {dx_out_mul}: relative L2 {rel:.3e}.  A '
            f'result near 1.0 means the odd-N half-pixel origin is not '
            f'being carried.')

    def test_the_faithful_zone_warning_sizes_each_axis_separately(self):
        """``resample_field``'s extent-preserving default gives each axis
        its own sample count, so the periodicity test has to as well --
        that is what ``_warn_mft_output_window(N_out_y=…)`` exists for.

        Oracle: the period is the input extent per axis (``N_in*dx_in``)
        and the window is the output extent per axis
        (``N_out_axis*dx_out``), both exact products of the arguments.
        Fixture: a 44x63 input at 1 um resampled to 0.7 um, where the
        extent-preserving default rounds y UP (63 samples x 0.7 um =
        44.1 um against a 44 um period, 1.0023x) and x exactly ON its
        period (90 x 0.7 = 63.0 um against 63 um).  So the warning must
        fire, and name y ALONE.

        Bar: the warning fires and its axis list is 'y'.  There is no
        tolerance here -- the comparison is between two exact products.

        The 45x63 companion is the half that bites: there y rounds DOWN
        (64 x 0.7 = 44.8 um inside a 45 um period) and x lands exactly on
        its period, so the call must be SILENT.

        FAIL-BEFORE: with the y axis sized by ``N_out`` -- the x count --
        the 45x63 call compares x's 63 um window against y's 45 um period
        and warns about an axis that fits, while the 44x63 call keeps
        warning for a reason that is no longer the measured one.
        """
        def _run(Ny, Nx, dx_in, dx_out, N_out=None):
            x = (np.arange(Nx) - Nx / 2) * dx_in
            y = (np.arange(Ny) - Ny / 2) * dx_in
            X, Y = np.meshgrid(x, y, indexing='xy')
            E = np.exp(-(X ** 2 + Y ** 2) / self.W0 ** 2).astype(complex)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                got, _ = resample_field(E, dx_in, dx_out, N_out,
                                        method='chirpz')
            return got, [str(w.message) for w in caught
                         if issubclass(w.category, UserWarning)]

        got, msgs = _run(44, 63, 1e-6, 0.7e-6)
        assert got.shape == (63, 90)                  # y rounds UP: 44.1 um
        assert msgs, (
            'a y window of 44.1 um against a 44 um period leaves the '
            'faithful zone and must warn')
        assert 'on y' in msgs[0] and 'on x' not in msgs[0], (
            f'the warning must name the y axis alone; got {msgs[0]!r}')

        got, msgs = _run(45, 63, 1e-6, 0.7e-6)
        assert got.shape == (64, 90)                  # y rounds DOWN: 44.8
        assert not msgs, (
            f'64 x 0.7 um = 44.8 um fits inside the 45 um y period and '
            f'90 x 0.7 um = 63.0 um lands exactly on the 63 um x period, '
            f'so this call is faithful on both axes; got {msgs!r}')

        got, msgs = _run(44, 63, 1e-6, 0.7e-6, N_out=40)
        assert not msgs, (
            '40 x 0.7 um = 28 um fits inside both periods (44 and 63 um)')
