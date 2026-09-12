"""WP-A2 regression pins for the opt-in refraction models of
``apply_real_lens`` -- the ``displaced`` remaps and their caches, the
screen-obliquity carrier units, and the ``tangent_facet_remap`` fold guards
(audit 2026-09-11, findings L3, L4, L5, L6, L7, L8, L9).

Split from ``test_audit2609_a2_analytic_lens.py`` because these paths run a
geometric ray trace per call and are an order slower than the screen tests.
Every bar carries its derivation, its oracle and the measured value on both
sides of the fix.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy.glass as _glass
from lumenairy.elements._lens_real import (
    _build_displaced_ray_map,
    _screen_obliquity_angle_field,
    apply_real_lens,
    clear_displaced_lut_cache,
    clear_pointwise_cos_grid_cache,
    set_pointwise_cos_grid_cache_budget,
)
from lumenairy.elements.lenses import surface_sag_general as _sag
from lumenairy.glass import get_glass_index


# ===========================================================================
# L3 -- the displaced remaps must CARRY the input phase
# ===========================================================================
class TestL3RemapsCarryTheInputPhase:
    """Both remap paths sampled only ``np.abs(E_in)`` and rebuilt the exit
    phase from the ``conjugate`` congruence (default: collimated), so every
    bit of phase the caller's field carried was discarded with no warning --
    and because ``displaced_obliquity='auto'`` routes any decentered / tilted
    / ``sag_callable`` element to the 2-D remap, that was the DEFAULT path for
    an asymmetric element.  Measured pre-fix: a flat and a 35-wave-defocused
    input produced outputs identical to 4.7e-16 of peak.
    """

    LAM = 0.55e-6
    N, DX, AP = 256, 1.2e-5, 2.4e-3

    def _inputs(self):
        k0 = 2.0 * np.pi / self.LAM
        ax = (np.arange(self.N) - self.N / 2) * self.DX
        X, Y = np.meshgrid(ax, ax)
        amp = np.exp(-(X ** 2 + Y ** 2) / (self.AP / 3.0) ** 2)
        W = (X ** 2 + Y ** 2) / (2 * 0.20) + 0.004 * X   # 200 mm defocus + tilt
        return (amp.astype(np.complex128),
                (amp * np.exp(1j * k0 * W)).astype(np.complex128),
                X, Y, float(np.ptp(W) / self.LAM))

    def _rx(self, dec=None):
        s0 = dict(radius=19.6e-3, glass_before='AIR', glass_after='N-BK7')
        s1 = dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR')
        if dec is not None:
            s0, s1 = dict(s0, decenter=dec), dict(s1, decenter=dec)
        return dict(surfaces=[s0, s1], thicknesses=[2.5e-3],
                    aperture_diameter=self.AP)

    @pytest.mark.parametrize('label,kw,dec', [
        ('1-D remap', dict(surface_model='displaced',
                           displaced_mode='remap'), None),
        ('2-D remap (default for an asymmetric element)',
         dict(surface_model='displaced'), (0.3e-3, 0.0)),
    ])
    def test_a_phased_input_gives_a_different_output(self, label, kw, dec):
        """DERIVATION OF THE BAR.  The two inputs differ by 35.4 waves p-v of
        phase over the pupil, which is a physically enormous difference: the
        THIN reference model separates them by 1.978 (normalised max|dE| /
        max|E|, i.e. ~2 because the two fields are anti-phased somewhere), and
        the pointwise SCREEN by 1.977.  Measured on the remaps after the fix:
        1.974 (1-D) and 1.961 (2-D), i.e. within 1 % of the reference models.
        Pre-fix both read 4.7e-16 -- the numerical zero.

        The bar is 0.5, a quarter of the reference separation and fifteen
        decades above the pre-fix value.  It cannot be met by anything that
        discards the phase.
        """
        E_flat, E_ph, X, Y, pv = self._inputs()
        assert pv > 30.0, "fixture lost its phase difference"
        rx = self._rx(dec)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            A = apply_real_lens(E_flat, prescription=rx,
                                wavelength=self.LAM, dx=self.DX, **kw)
            B = apply_real_lens(E_ph, prescription=rx,
                                wavelength=self.LAM, dx=self.DX, **kw)
        m = (X ** 2 + Y ** 2) <= (0.8 * self.AP / 2) ** 2
        rel = float(np.max(np.abs(A - B)[m])
                    / max(float(np.max(np.abs(A)[m])), 1e-30))
        assert rel > 0.5, (
            f"{label}: a {pv:.1f}-wave phase difference at the input changed "
            f"the output by {rel:.3e} of peak.  The remap is rebuilding the "
            f"exit phase from the congruence and discarding the field's own.")

    def test_a_diverging_source_focuses_where_it_should(self):
        """The physical consequence, measured by a through-focus scan.

        DERIVATION OF THE BAR.  A 150 mm diverging source through this singlet
        (f = 22.1 mm) images at 1/(1/f - 1/150 mm) = 25.9 mm; the COLLIMATED
        focus is 22.1 mm.  Both the thin model and the pointwise screen put
        the best focus at 25.0 mm on this grid.  Pre-fix the 2-D remap with
        the DEFAULT ``conjugate=None`` reported 21.0 mm -- the collimated
        answer, 4 mm / 19 % out -- because it replaced the input's divergence
        with the congruence it had traced.  Measured after the fix: 25.0 mm,
        equal to the thin reference.

        The bar is that the remap must land within one scan step (1 mm) of the
        THIN model on the SAME grid, which is a peer-model agreement rather
        than an absolute-focus claim, and the 4 mm defect is four steps away.
        """
        from lumenairy.propagators.propagation import (
            angular_spectrum_propagate as asm,
        )
        k0 = 2.0 * np.pi / self.LAM
        ax = (np.arange(self.N) - self.N / 2) * self.DX
        X, Y = np.meshgrid(ax, ax)
        amp = np.exp(-(X ** 2 + Y ** 2) / (self.AP / 3.0) ** 2)
        s = 0.150
        E_in = (amp * np.exp(1j * k0 * (X ** 2 + Y ** 2) / (2 * s))
                ).astype(np.complex128)
        rx = self._rx((0.3e-3, 0.0))
        zs = np.arange(19.0, 30.1, 1.0) * 1e-3

        def _best(kw):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                Eo = apply_real_lens(E_in, prescription=rx,
                                     wavelength=self.LAM, dx=self.DX, **kw)
                pk = [float(np.abs(asm(Eo.copy(), z, self.LAM, self.DX)).max())
                      for z in zs]
            return float(zs[int(np.argmax(pk))])

        z_thin = _best({})
        z_remap = _best(dict(surface_model='displaced'))
        assert abs(z_remap - z_thin) <= 1.0e-3, (
            f"the 2-D remap focuses a 150 mm diverging source at "
            f"{z_remap * 1e3:.1f} mm where the thin model puts it at "
            f"{z_thin * 1e3:.1f} mm.  Pre-fix the remap reported the "
            f"COLLIMATED focus (21 mm) because it discarded the input "
            f"curvature.")


class TestL9RemapLatticeSmoothingIsAnnounced:
    """The 2-D remap rebuilds the whole exit field from a fixed 181x181 launch
    lattice, so input structure finer than the LAUNCH pitch -- a hard stop
    edge, an obscuration, an upstream DOE, speckle -- is not propagated, it is
    smoothed away.  Measured by the audit: a ripple at 2.2 launch samples per
    period comes back at 0.51 of its input contrast where the field-grid screen
    path resolves it at 1.26.  Nothing said so.

    Raising the lattice is NOT the fix and is deliberately not done: scored by
    the mirror symmetry of the image-plane intensity (an exact symmetry of the
    physics), the Delaunay backend reads 7.9e-14 at n_side = 181 and 4.1e-03 at
    512 -- the denser scattered set is not reflection-stable.  So the finding's
    SILENCE is what is fixed, and this pins the announcement.
    """

    LAM = 0.55e-6

    @staticmethod
    def _rx(dec):
        return dict(surfaces=[dict(radius=19.6e-3, glass_before='AIR',
                                   glass_after='N-BK7', decenter=dec),
                              dict(radius=-27.4e-3, glass_before='N-BK7',
                                   glass_after='AIR', decenter=dec)],
                    thicknesses=[2.5e-3], aperture_diameter=2.0e-3)

    def _warns(self, N, dx):
        ax = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(ax, ax)
        E = np.exp(-(X ** 2 + Y ** 2) / (0.67e-3) ** 2).astype(np.complex128)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            apply_real_lens(E, prescription=self._rx((0.3e-3, 0.0)),
                            wavelength=self.LAM, dx=dx,
                            surface_model='displaced')
        return [str(x.message) for x in w
                if 'launch lattice' in str(x.message)]

    def test_a_finely_sampled_grid_is_told_the_remap_smooths(self):
        """2 mm aperture on a 2 um pitch: the launch lattice is 11.1 um, 5.6x
        coarser than the field, and the audit's ripple fixture lives exactly
        there.  The message must name both pitches and an alternative."""
        msgs = self._warns(2048, 2e-6)
        assert msgs, "the remap smoothed silently on a 2 um grid"
        m = msgs[0]
        assert 'um pitch' in m and 'field pitch' in m
        assert 'pointwise' in m and 'apply_real_lens_traced' in m

    def test_a_coarse_grid_is_not_nagged(self):
        """At a field pitch the lattice already resolves there is nothing to
        announce -- the guard must not fire on every displaced call."""
        assert not self._warns(128, 32e-6)


# ===========================================================================
# L4 -- carrier units in _screen_obliquity_angle_field
# ===========================================================================
class TestL4CarrierMomentumUnits:
    """``_screen_obliquity_angle_field`` must return the transverse OPTICAL
    momentum ``q``.  Two of its four carrier vocabularies deliver DIRECTION
    COSINES and need the ``n1`` multiply; the other two deliver ``q`` already
    and must not be scaled.  Multiplying those by ``n1`` over-counted by
    exactly ``n1`` -- x1.5168 in N-BK7 -- which made the "corrected" screen
    WORSE than the uncorrected one at 100 mrad (0.0971 vs 0.0784 waves).

    Oracle: an exact plane wave built IN the glass at a known in-glass angle,
    so the true ``q = n1 sin(theta)`` is known in closed form (error floor
    zero).  Bar 2e-4 absolute on ``q = 0.0758``: the 'auto' branch reads the
    field's phase through a finite-difference fit whose own error at 50 mrad
    and this sampling is ~1e-5, and the defect is +0.039 (a 200x margin).
    """

    LAM = 0.55e-6
    N, DX, N1, THETA = 512, 1.0e-6, 1.5168, 0.050

    def _field(self):
        k0 = 2.0 * np.pi / self.LAM
        ax = (np.arange(self.N) - self.N / 2) * self.DX
        X, Y = np.meshgrid(ax, ax)
        p = self.N1 * np.sin(self.THETA)
        return (np.exp(-(X * X + Y * Y) / (120e-6) ** 2)
                * np.exp(1j * k0 * p * X)).astype(np.complex128), X, p

    def _q(self, carrier, E):
        qx, qy = _screen_obliquity_angle_field(
            carrier, E, self.LAM, self.DX, self.DX, self.N, self.N,
            n_medium=self.N1)
        return float(qx) if np.ndim(qx) == 0 else float(np.median(qx))

    def test_geometric_carriers_are_scaled_by_n1(self):
        from lumenairy.elements._lens_traced import TiltedCarrier
        E, X, p = self._field()
        got = self._q(TiltedCarrier(L=np.sin(self.THETA), M=0.0, R=np.inf), E)
        assert got == pytest.approx(p, abs=2e-4), (
            "a TiltedCarrier states DIRECTION COSINES; the consumer needs "
            "n1 * (L, M)")

    def test_auto_and_ndarray_carriers_are_not_scaled(self):
        E, X, p = self._field()
        for label, carrier in (('auto', 'auto'), ('ndarray W', p * X)):
            got = self._q(carrier, E)
            assert got == pytest.approx(p, abs=2e-4), (
                f"carrier={label!r} already delivers OPTICAL momentum "
                f"(it is read from k0*S with S the optical path); got "
                f"{got:.6f} against {p:.6f} -- a factor "
                f"{got / p:.4f}, which is n1 double-counted.")

    def test_a_scalar_conjugate_is_scaled_by_n1(self):
        """``W = sign(s)(sqrt(x^2+y^2+s^2) - |s|)`` is a geometric distance, so
        ``grad W = sin alpha`` is a direction cosine.  Checked at the edge of
        a 2 mm radius where alpha is large enough to measure."""
        E, X, p = self._field()
        s = 0.05
        r = 100 * self.DX
        expect = self.N1 * r / np.sqrt(r * r + s * s)
        qx, _ = _screen_obliquity_angle_field(
            s, E, self.LAM, self.DX, self.DX, self.N, self.N,
            n_medium=self.N1)
        j = self.N // 2 + 100
        assert float(qx[self.N // 2, j]) == pytest.approx(expect, rel=1e-6)


# ===========================================================================
# L5 / L6 -- cache keys must be VALUE keys
# ===========================================================================
class TestL5L6CacheKeyCompleteness:
    LAM = 1.31e-6
    N, DX, AP = 128, 1.6e-5, 1.2e-3

    @pytest.fixture(autouse=True)
    def _clean_caches(self):
        clear_displaced_lut_cache()
        clear_pointwise_cos_grid_cache()
        yield
        set_pointwise_cos_grid_cache_budget(0)
        clear_pointwise_cos_grid_cache()
        clear_displaced_lut_cache()
        _glass.GLASS_REGISTRY.pop('WPA2GLASS', None)
        _glass._clear_glass_caches()

    def test_a_mutated_sag_callable_is_a_miss_not_a_stale_hit(self):
        """L5.  ``_DISPLACED_COS_GRID_CACHE`` keyed a freeform ``sag_callable``
        by OBJECT IDENTITY, which does not imply value equality for a MUTABLE
        callable -- and the cache is sold for exactly the workload that mutates
        one ("a decentered-design iteration loop that only moves the field
        re-uses the ~3.9 s trace").  Measured pre-fix: 164 % of peak amplitude
        of error from the stale grid.

        Two-sided and tolerance-free: the cached answer must equal the COLD
        answer for the mutated state, byte for byte.  That is the definition of
        a correct memo, and it is what identity keying could not deliver.
        """
        class FF:
            def __init__(self, a):
                self.a = a

            def __call__(self, x, y):
                return self.a * (x * x + y * y)

        cb = FF(5.0)
        rx = dict(surfaces=[dict(radius=25e-3, glass_before='AIR',
                                 glass_after='N-BK7', sag_callable=cb),
                            dict(radius=-25e-3, glass_before='N-BK7',
                                 glass_after='AIR')],
                  thicknesses=[2e-3], aperture_diameter=self.AP)
        E = np.ones((self.N, self.N), dtype=np.complex128)
        kw = dict(wavelength=self.LAM, dx=self.DX, surface_model='displaced',
                  displaced_obliquity='pointwise')
        set_pointwise_cos_grid_cache_budget(64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens(E, prescription=rx, **kw)       # populate
            cb.a = -5000.0
            cached = apply_real_lens(E, prescription=rx, **kw)
            clear_pointwise_cos_grid_cache()
            cold = apply_real_lens(E, prescription=rx, **kw)
        assert np.array_equal(cached.view(np.uint8), cold.view(np.uint8)), (
            f"the cos-grid cache returned a grid traced against the callable's "
            f"OLD state: max|d| = {np.max(np.abs(cached - cold)):.4e} against "
            f"a peak of {np.max(np.abs(cold)):.4e}")

    def test_repointing_a_glass_registry_entry_is_a_miss(self):
        """L6.  ``_DISPLACED_LUT_CACHE`` (ON by default) keyed glasses by
        registry NAME.  ``GLASS_REGISTRY`` is a documented mutable extension
        point, so re-pointing an entry under the same name left the key
        unchanged and the cached obliquity cosines stale -- measured 1 % of
        peak amplitude.  Keying on the RESOLVED index removes the hazard.

        Two-sided: the cached answer must equal the cold one byte for byte.
        """
        _glass.GLASS_REGISTRY['WPA2GLASS'] = lambda wl: 1.50
        _glass._clear_glass_caches()
        rx = dict(surfaces=[dict(radius=25e-3, glass_before='AIR',
                                 glass_after='WPA2GLASS'),
                            dict(radius=-25e-3, glass_before='WPA2GLASS',
                                 glass_after='AIR')],
                  thicknesses=[2e-3], aperture_diameter=self.AP)
        E = np.ones((self.N, self.N), dtype=np.complex128)
        kw = dict(wavelength=self.LAM, dx=self.DX, surface_model='displaced')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens(E, prescription=rx, **kw)       # populate the LUT
            _glass.GLASS_REGISTRY['WPA2GLASS'] = lambda wl: 1.90
            _glass._clear_glass_caches()                    # index cache only
            cached = apply_real_lens(E, prescription=rx, **kw)
            clear_displaced_lut_cache()
            cold = apply_real_lens(E, prescription=rx, **kw)
        assert np.array_equal(cached.view(np.uint8), cold.view(np.uint8)), (
            f"the displaced LUT cache kept cosines traced against the OLD "
            f"index for a re-pointed registry name: max|d| = "
            f"{np.max(np.abs(cached - cold)):.4e}")


# ===========================================================================
# L7 -- the exit referencing leg is not always in air
# ===========================================================================
class TestL7DisplacedRemapExitIndex:
    def test_exit_leg_uses_the_last_surfaces_glass_after(self):
        """DERIVATION OF THE BAR.  Fixture: ONE curved face, AIR -> N-BK7, with
        no gap, so the exit vertex plane is z = 0 and every ray must be walked
        BACK through N-BK7 from its landing point at z = sag(h).  The whole
        trace is then closed form:

            OPL(h) = 1.0 * sag(h)            (entrance leg, in air)
                   + n2 * (0 - sag(h)) / Lz  (exit leg, in the GLASS)

        with ``Lz = cos(theta_i - theta_t)`` from Snell at the facet.  Error
        floor: exact.

        Measured: max|model - closed form| = 8.1e-19 m with the correct exit
        index, and 1.03e-5 m (16.3 waves at 632.8 nm) with the hard-coded
        ``n = 1`` -- thirteen decades apart.  The bar is 1e-12 m: six decades
        above the float64 floor of a ~1e-5 m quantity, seven below the defect.
        """
        lam = 632.8e-9
        R = 25e-3
        n2 = float(get_glass_index('N-BK7', lam))
        rx_s = [dict(radius=R, glass_before='AIR', glass_after='N-BK7')]
        r_max = 1.0e-3
        h_in, h_out, opl = _build_displaced_ray_map(
            rx_s, [], lam, r_max, n_fan=257)
        sag = _sag(h_in * h_in, R, 0.0, None)
        # Snell at the facet; the refracted ray makes (th_i - th_t) with z.
        c = 1.0 / R
        dsag = c * h_in / np.sqrt(1.0 - c * c * h_in * h_in)
        th_i = np.arctan(dsag)
        th_t = np.arcsin(np.sin(th_i) / n2)
        Lz = np.cos(th_i - th_t)
        expect = 1.0 * sag + n2 * (0.0 - sag) / Lz
        expect = expect - expect[0]
        got = opl - opl[0]
        wrong = (1.0 * sag + 1.0 * (0.0 - sag) / Lz)
        wrong = wrong - wrong[0]
        err = float(np.max(np.abs(got - expect)))
        err_if_air = float(np.max(np.abs(wrong - expect)))
        assert err_if_air > 1e-6, "fixture has no immersed-exit signal"
        assert err < 1e-12, (
            f"the displaced ray map's exit leg is off by {err:.3e} m; with the "
            f"exit index hard-coded to 1 it would be {err_if_air:.3e} m "
            f"({err_if_air / lam:.1f} waves).")


# ===========================================================================
# L8 -- the tangent_facet_remap guards must score the ILLUMINATED support
# ===========================================================================
class TestL8RemapGuardsScoreTheSupport:
    """``det(I + dW/dx)`` and the pull-back residual were reduced over EVERY
    pixel of the grid, including the padding outside the clear aperture where
    the field is exactly zero and where ``sag`` and ``grad sag`` grow without
    bound.  A converging beam needs a padded grid, so padding CAUSED refusals:
    an 8x pad was declined at ``min det = -0.82`` on dark corner pixels while
    the illuminated pupil sat at 0.9986, three orders inside the 1e-4 bar --
    and the message blamed the physics ("use a different model") when the
    remedy was to shrink the grid.
    """

    LAM = 0.55e-6

    @staticmethod
    def _rx():
        return dict(surfaces=[dict(radius=19.6e-3, glass_before='AIR',
                                   glass_after='N-BK7'),
                              dict(radius=-27.4e-3, glass_before='N-BK7',
                                   glass_after='AIR')],
                    thicknesses=[2.5e-3], aperture_diameter=2.0e-3)

    def test_an_eight_times_padded_grid_is_accepted(self):
        """Two-sided in the only sense a refusal admits: the call must RETURN,
        with a finite field carrying the beam's power, on a grid whose window
        is 8.2x the 2 mm pupil.  Pre-fix this raised ``the transverse-walk map
        folds.  min det = -0.821741``, scored on corner pixels the beam never
        reaches.  Energy is checked too so that "accepted" cannot be met by
        returning zeros.
        """
        N, dx = 2048, 8e-6
        ax = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(ax, ax)
        E = np.exp(-(X ** 2 + Y ** 2) / (0.67e-3) ** 2).astype(np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = apply_real_lens(E, prescription=self._rx(),
                                  wavelength=self.LAM, dx=dx,
                                  surface_model='tangent_facet_remap')
        assert np.all(np.isfinite(out))
        p_in = float(np.sum(np.abs(E) ** 2))
        p_out = float(np.sum(np.abs(out) ** 2))
        # The entrance aperture clips the Gaussian at r = 1 mm; the model then
        # conserves energy to its Jacobian's accuracy (measured 0.99998).
        assert 0.5 * p_in < p_out < 1.01 * p_in, (
            f"the remap returned {p_out / p_in:.4f} of the input power")

    def test_a_genuine_fold_inside_the_pupil_is_still_refused(self):
        """The guard must still decline what it exists to decline.  A fold
        inside the ILLUMINATED support is unrepresentable by any pull-back, so
        an f/0.6-class element at a 4 mm pupil must raise -- and the message
        must say which reduction fired.
        """
        rx = dict(surfaces=[dict(radius=4.0e-3, glass_before='AIR',
                                 glass_after='N-SF11'),
                            dict(radius=-4.0e-3, glass_before='N-SF11',
                                 glass_after='AIR')],
                  thicknesses=[6e-3], aperture_diameter=6.0e-3)
        N, dx = 512, 8e-6
        ax = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(ax, ax)
        E = np.exp(-(X ** 2 + Y ** 2) / (2.0e-3) ** 2).astype(np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ValueError, match=r"REFUSES at surface"):
                apply_real_lens(E, prescription=rx, wavelength=self.LAM,
                                dx=dx, surface_model='tangent_facet_remap')
