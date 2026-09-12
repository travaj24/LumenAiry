"""v5.2 -- off-axis conic in surface frame (ROADMAP v5.1 deferred).

Pins the ``surface_frame=True`` opt-in branch added to
:func:`lumenairy.elements._lens_real.apply_real_lens` per the
ROADMAP entry "Off-axis conic in surface frame (not just
decenter+tilt)":

    "apply_real_lens honours decenter and tilt keys but adds them to
    the field's coordinates, not the surface's frame.  Tilted /
    displaced asphere with proper sag in surface frame requires a
    coordinate transformation Optiland and Zemax do natively."

The v5.2 opt-in is ``apply_real_lens(..., surface_frame=True)``.
Default ``surface_frame=False`` preserves the v5.1 field-frame
behaviour bit-for-bit (covered by the backward-compat pin below).

Tests
-----
* ``test_parabola_decenter_only_paraxial_pin``: a parabola
  (``R=10mm``, ``conic=-1``) decentered by ``1mm`` in x, no tilt.
  In the surface-frame branch the sag at the field-origin equals the
  parabolic sag at ``r = 1mm`` (paraxial limit ``r**2 / (2R)``).
  Verifies the per-pixel OPD on the central column matches the
  closed-form parabolic sag profile.

* ``TestTiltedSurfaceDeviatesTheBeam``: a tilted refracting face is a thin
  prism, so it must deviate the beam by ``(n2 - n1) * theta`` -- in BOTH
  branches and about the SAME axis.  Read from the exit field's spectral
  centroid, i.e. from where the beam actually goes.  (This replaces a pin
  that asserted the surface-frame branch had NO tilt ramp, which was the
  defect: see the class docstring.)

* ``TestFieldFrameTiltRampConvention``: the field-frame branch's
  ``tilt = (t0, t1)`` IS the sag ramp ``t0*x + t1*y``, pinned against that
  closed form; and a pure decenter is bit-identical in both frames.  (This
  replaces two ``array_equal(no_kwarg, default_kwarg)`` assertions that
  tested CPython's default-argument mechanism rather than this library.)

* ``test_optiland_cross_check_smoke``: optional cross-check.  Skips
  if Optiland is not importable.  Confirms the v5.2 branch is
  callable without raising on an Optiland-style off-axis-conic
  prescription.
"""

from __future__ import annotations

import importlib.util as _ilu

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.elements._lens_real import apply_real_lens


def _flat_singlet_prescription(R, conic=0.0, decenter=(0.0, 0.0),
                                tilt=(0.0, 0.0), aperture=8e-3):
    """One-surface refractor in air (n=1 -> n=1) with an explicit
    decenter / tilt on the surface.  ``n2 - n1 = 0`` would zero the
    OPD; we use a fictitious 'glass' lookup so the per-surface phase
    actually carries the sag.  Easiest: borrow N-BK7 for the after
    glass and rely on a tiny thickness so the in-glass propagation
    is essentially identity.

    This helper is intentionally minimal -- the tests below assert
    on the SAG that gets baked into the phase screen, not on the
    propagation through glass.
    """
    return {
        'aperture_diameter': aperture,
        'surfaces': [
            {'radius': R, 'conic': conic,
             'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': 'air', 'glass_after': 'N-BK7',
             'decenter': decenter, 'tilt': tilt},
            {'radius': float('inf'), 'conic': 0.0,
             'aspheric_coeffs': None,
             'radius_y': None, 'conic_y': None,
             'aspheric_coeffs_y': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [1e-6],  # essentially nil glass
    }


def _extract_opd(E_in, E_out, wavelength):
    """Recover the per-pixel OPD imprinted between two complex fields
    via the unwrapped phase difference.  Used as a diagnostic in the
    tests below."""
    phase_diff = np.angle(E_out * np.conj(E_in))
    return -phase_diff * wavelength / (2.0 * np.pi)


class TestParabolaDecenterOnlySurfaceFramePin:
    """Closed-form pin: a paraxial-sag parabola decentered by
    ``decenter_x``, no tilt.  In the surface-frame branch the sag
    at the field-origin equals the parabolic sag evaluated at
    ``r_s = decenter_x`` (the field origin maps to the surface-
    frame point ``(-decenter_x, 0)``).

    Paraxial limit: ``sag(r) = r**2 / (2R)``, so the OPD difference
    between the field origin and the field point at ``+decenter_x``
    is ``(n2 - n1) * decenter_x**2 / (2R)``.

    Sized to keep the OPD comfortably below lambda/4 so phase
    wrap does not corrupt ``np.angle``-based OPD extraction, with
    the grid extent comfortably exceeding the aperture.
    """

    def test_paraxial_sag_at_field_origin(self):
        N, dx, wl = 256, 4e-6, 1.55e-6
        R = 10e-3                  # 10 mm paraxial radius
        decenter_x = 1e-4          # 0.1 mm decenter (small enough to
                                   # keep OPD << lambda/4)
        aperture = 0.8e-3          # 0.8 mm aperture (< N*dx = 1.024 mm)

        E_in = np.ones((N, N), dtype=np.complex128)
        prescription = _flat_singlet_prescription(
            R=R, conic=-1.0,                # parabola
            decenter=(decenter_x, 0.0), tilt=(0.0, 0.0),
            aperture=aperture)
        # Use the default thin-element OPD so the comparison against
        # ``(n2-n1)*sag`` is exact, with bandlimit=False to avoid
        # ASM band-limit artefacts on the tiny 1 um propagation.
        E_out = apply_real_lens(
            E_in, prescription=prescription,
            wavelength=wl, dx=dx, bandlimit=False,
            surface_frame=True,
        )

        # Field-origin pixel index for an even-N grid.
        i_mid = N // 2
        opd_map = _extract_opd(E_in, E_out, wl)
        opd_origin = opd_map[i_mid, i_mid]

        # Closed-form expected OPD at the field origin (paraxial
        # parabola): conic=-1 + R=10mm gives sag(r) = r^2 / (2R).
        # The surface-frame inverse transform maps the field origin
        # to (-decenter_x, 0) in surface coords.
        from lumenairy.glass import get_glass_index
        n1 = get_glass_index('air', wl)
        n2 = get_glass_index('N-BK7', wl)
        sag_expected = (decenter_x ** 2) / (2.0 * R)
        opd_expected = (n2 - n1) * sag_expected

        # Reference to the field pixel at +decenter_x: its surface-
        # frame coord is (0, 0), so sag = 0 there.
        i_at_decenter = int(np.argmin(
            np.abs((np.arange(N) - N / 2) * dx - decenter_x)))
        opd_at_decenter = opd_map[i_mid, i_at_decenter]

        # Sanity-check: keep the OPD well under lambda/4 so phase
        # unwrap is not an issue (lambda/4 ~ 388 nm at 1.55 um;
        # opd_expected ~ 250 pm for these numbers).
        assert abs(opd_expected) < wl / 4.0, (
            f"Test mis-sized: expected OPD {opd_expected:.3e} m "
            f"exceeds lambda/4 = {wl/4.0:.3e} m; phase unwrap will "
            f"corrupt the comparison.  Shrink decenter_x."
        )

        # The OPD DIFFERENCE between the field origin and the
        # field-point at +decenter_x equals (n2-n1)*sag_expected
        # because the in-glass piston cancels.  Tolerance is set
        # to a few percent to allow for the small ASM through-glass
        # diffraction residual and grid-discretisation error in
        # locating the +decenter_x pixel.
        opd_diff = opd_origin - opd_at_decenter
        assert opd_diff == pytest.approx(opd_expected, rel=0.05), (
            f"v5.2 surface-frame OPD pin failed: expected closed-form "
            f"OPD difference {opd_expected:.6e} m for parabola "
            f"R={R}, decenter={decenter_x} -- got {opd_diff:.6e} m."
        )


class TestTiltedSurfaceDeviatesTheBeam:
    """A tilted refracting face is a thin prism: it must deviate the beam by
    ``(n2 - n1) * theta``, in BOTH the field-frame and the surface-frame
    branch, and about the SAME axis.

    WHAT THIS CLASS USED TO ASSERT, AND WHY IT WAS THE DEFECT (audit
    2026-09-11, finding V2 / L2).  ``test_tilted_parabola_branches_differ``
    asserted that under ``surface_frame=True`` the OPD's x-slope was under
    20 % of ``(n2-n1)*tx`` -- "the ramp is gone, only the parabola's residual
    sag-vs-x remains".  The premise is wrong: a rigid-body rotation
    RE-EXPRESSES the tilt ramp, it does not delete it.  The branch evaluated
    the sag at the rotated transverse footprint and discarded the rotated
    surface's own field-frame HEIGHT, which is where the tilt lives -- so a
    tilted FLAT face was a byte-identical no-op (deviation exactly 0.000 mrad
    where a thin prism gives 2.575 mrad), and a tilted R = 50 mm sphere lost
    8.15 waves of OPD at 5 mrad over a +-2 mm pupil.  The test passed
    throughout because it asserted the missing term was missing, and its
    companion "the branches differ" assertion passed only on the second-order
    ``cos(tx)`` scaling of the y-arm -- on a FLAT face the branch was provably
    a no-op and nothing here would have noticed.

    The oracle below is the thin-prism deviation, which is exact for a flat
    face and needs nothing from this library: a plane wave crossing a wedge of
    small angle ``theta`` between media ``n1`` and ``n2`` leaves at
    ``(n2 - n1) * theta`` to the axis.  The deviation is read from the exit
    field's own spectral centroid, i.e. from where the beam GOES, which is the
    quantity the old test never looked at.
    """

    N, DX, WL = 256, 6e-6, 1.0e-6
    THETA = 5e-3

    @classmethod
    def _plate(cls, tilt):
        """N-BK7 plate whose FIRST face carries the tilt.  No
        ``aperture_diameter``: the Gaussian is the stop, so the spectral
        centroid reads the deviation and not a truncation."""
        return {
            'surfaces': [
                {'radius': float('inf'), 'conic': 0.0,
                 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'tilt': tilt},
                {'radius': float('inf'), 'conic': 0.0,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [2e-3],
        }

    @classmethod
    def _deviation(cls, tilt, surface_frame):
        """(sin theta_x, sin theta_y) of the exit beam, from the power-weighted
        centroid of its angular spectrum."""
        N, dx, wl = cls.N, cls.DX, cls.WL
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        E_in = np.exp(-(X ** 2 + Y ** 2) / (0.4e-3) ** 2).astype(np.complex128)
        E_out = apply_real_lens(
            E_in, prescription=cls._plate(tilt), wavelength=wl, dx=dx,
            surface_frame=surface_frame)
        fx = np.fft.fftfreq(N, dx)
        FX, FY = np.meshgrid(fx, fx)
        S = np.abs(np.fft.fft2(E_out)) ** 2
        tot = float(S.sum())
        return (float((S * FX).sum()) / tot * wl,
                float((S * FY).sum()) / tot * wl)

    def test_tilted_flat_face_deviates_by_n_minus_one_theta(self):
        """DERIVATION OF THE BAR.  Oracle: the thin-prism deviation
        ``(n-1)*theta`` = 2.575e-3 rad for N-BK7 at 1 um and theta = 5 mrad.
        Its own error is the small-angle expansion, O(theta**3) ~ 1e-7 of the
        value, plus the spectral-centroid readout on a 256-point grid
        (dominated by the Gaussian's own spectral width, which is symmetric
        and cancels in the centroid to ~1e-5 relative).  Measured: field frame
        -2.5744e-3, surface frame -2.5744e-3 -- both 0.04 % from the oracle.
        The bar is 2 %: three decades above the oracle floor and 50x below the
        100 % error the defect produced (surface frame gave exactly 0.000).
        """
        from lumenairy.glass import get_glass_index
        n = float(get_glass_index('N-BK7', self.WL))
        expected = (n - 1.0) * self.THETA
        for sf in (False, True):
            sx, sy = self._deviation((self.THETA, 0.0), sf)
            assert abs(abs(sx) - expected) < 0.02 * expected, (
                f"surface_frame={sf}: a flat face tilted {self.THETA * 1e3:g} "
                f"mrad deviated the beam by {abs(sx):.6e} rad in x, against "
                f"the thin-prism value (n-1)*theta = {expected:.6e}.  "
                f"surface_frame=True used to give exactly 0.")
            assert abs(sy) < 0.02 * expected, (
                f"surface_frame={sf}: a tilt in the first component deviated "
                f"the beam in y by {sy:.3e} rad; it must act in x.")

    def test_both_branches_deviate_about_the_same_axis(self):
        """The two branches must agree on WHICH AXIS a ``tilt`` component acts
        about, or flipping ``surface_frame`` silently re-points the element.

        DERIVATION OF THE BAR.  Both branches read the same key and, for a
        flat face, both reduce to the same thin prism, so the two deviations
        are the same number to the readout floor: measured 4.3e-8 rad of
        difference against a 2.575e-3 rad deviation (1.7e-5 relative).  The
        bar is 2 % of the deviation -- 1000x the measured difference, and it
        fails outright on an axis swap (which makes the difference equal to
        the deviation itself, 100 %).
        """
        for tilt in ((self.THETA, 0.0), (0.0, self.THETA)):
            ff = np.array(self._deviation(tilt, False))
            sf = np.array(self._deviation(tilt, True))
            dev = float(np.hypot(*ff))
            assert dev > 1e-4, "fixture stopped deviating the beam"
            assert np.max(np.abs(ff - sf)) < 0.02 * dev, (
                f"tilt={tilt}: field frame deviates by {ff}, surface frame by "
                f"{sf}.  The two branches disagree about the tilt axis, so "
                f"flipping surface_frame re-points the element.")


class TestFieldFrameTiltRampConvention:
    """The FIELD-frame branch's tilt convention, pinned against a closed form.

    WHAT THIS CLASS REPLACES (audit 2026-09-11, finding V2).  Two tests here
    asserted ``np.array_equal(apply_real_lens(...),
    apply_real_lens(..., surface_frame=False))`` -- i.e. that passing a kwarg
    at its default equals omitting it.  That is a property of CPython's
    default-argument mechanism, not of this library: no change to the
    surface-frame branch, the tilt convention or the sag could make either
    assertion fail, and they were counted as the backward-compatibility pin.

    The substance they were reaching for -- "the field-frame branch's numbers
    do not move" -- is pinned here against an INDEPENDENT closed form instead.
    ``tilt = (t0, t1)`` on a flat face deposits the sag ramp ``t0*x + t1*y``,
    so the OPD is ``(n2 - n1) * (t0*x + t1*y)`` exactly.  This is the
    convention ``raytrace``'s ``field_tilt``, ``_disp_surface_z_grad`` and the
    lumenairy-free geometric spot oracle all share; it is also the one the
    surface-frame branch now matches.
    """

    def test_flat_face_tilt_ramp_matches_closed_form(self):
        """DERIVATION OF THE BAR.  Oracle: ``OPD = (n2-n1)*(t0*x + t1*y)``,
        exact for a flat face (there is no sag to add), so the oracle's own
        error is zero.

        The measurement differences the TILTED run against an otherwise
        identical UNTILTED one, ``angle(E_tilt * conj(E_flat))``, so the
        plate's own ``n*k0*t`` piston -- which is many waves and would wrap --
        cancels exactly and only the ramp is left.  The tilts are sized so the
        ramp spans 0.118 waves p-v over the scored disc, comfortably inside one
        wrap, so no unwrapping is involved either.

        Measured max deviation 6.7e-18 m against a 1e-12 m bar.  1 pm is six
        decades above the floating-point floor and six decades BELOW the
        defect it guards: a swapped or sign-flipped tilt component moves the
        OPD by 100 % of the 1.2e-7 m ramp.
        """
        from lumenairy.glass import get_glass_index
        N, dx, wl = 128, 5e-6, 1.0e-6
        t0, t1 = 3.0e-4, -5.0e-4

        def _rx(tilt):
            return {
                'surfaces': [
                    {'radius': float('inf'), 'conic': 0.0,
                     'glass_before': 'air', 'glass_after': 'N-BK7',
                     'tilt': tilt},
                    {'radius': float('inf'), 'conic': 0.0,
                     'glass_before': 'N-BK7', 'glass_after': 'air'},
                ],
                'thicknesses': [1e-6],
            }

        E_in = np.ones((N, N), dtype=np.complex128)
        kw = dict(wavelength=wl, dx=dx, bandlimit=False, surface_frame=False)
        E_t = apply_real_lens(E_in, prescription=_rx((t0, t1)), **kw)
        E_0 = apply_real_lens(E_in, prescription=_rx((0.0, 0.0)), **kw)
        # OPD of the tilt alone: screen is exp(-i k0 OPD), so
        # angle(E_t / E_0) = -k0 * OPD_ramp.
        opd = -np.angle(E_t * np.conj(E_0)) * wl / (2.0 * np.pi)
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        n = float(get_glass_index('N-BK7', wl))
        expect = (n - 1.0) * (t0 * X + t1 * Y)
        m = (X ** 2 + Y ** 2) <= (0.2e-3) ** 2
        assert np.ptp(expect[m]) / wl < 0.5, "fixture ramp would wrap"
        d = (opd - expect)[m]
        d = d - d.mean()
        assert np.max(np.abs(d)) < 1e-12, (
            f"field-frame tilt ramp deviates from (n2-n1)*(t0*x + t1*y) by "
            f"{np.max(np.abs(d)):.3e} m over a {np.ptp(expect[m]):.3e} m ramp; "
            f"the shipped convention is that tilt=(t0, t1) IS that ramp (and "
            f"raytrace's field_tilt reads the same key the same way).")

    def test_decenter_only_is_identical_in_both_frames(self):
        """A pure decenter is a translation, so the rigid-body and field-frame
        readings coincide EXACTLY -- a two-sided statement with no tolerance.

        This is the part of the old backward-compat pin that was real: it is
        what tells you the surface-frame branch's rotation machinery is inert
        when there is no rotation.  Unlike the assertions it replaces, it can
        fail: any change to the inverse rigid-body map that is not the
        identity at zero tilt breaks it.
        """
        N, dx, wl = 96, 8e-6, 1.55e-6
        rx = {
            'aperture_diameter': 0.7e-3,
            'surfaces': [
                {'radius': 50e-3, 'conic': 0.0,
                 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'decenter': (0.5e-3, -0.3e-3), 'tilt': (0.0, 0.0)},
                {'radius': -50e-3, 'conic': 0.0,
                 'glass_before': 'N-BK7', 'glass_after': 'air'},
            ],
            'thicknesses': [3e-3],
        }
        rng = np.random.default_rng(42)
        E_in = (rng.standard_normal((N, N))
                + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        a = apply_real_lens(E_in, prescription=rx, wavelength=wl, dx=dx,
                            bandlimit=True, surface_frame=False)
        b = apply_real_lens(E_in, prescription=rx, wavelength=wl, dx=dx,
                            bandlimit=True, surface_frame=True)
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), (
            f"decenter-only: the two frames must agree bit for bit; "
            f"max|d| = {np.max(np.abs(a - b)):.3e}")


@pytest.mark.skipif(
    _ilu.find_spec('optiland') is None,
    reason="Optiland not installed; skipping cross-check.",
)
class TestOptilandCrossCheckSmoke:
    """Optional cross-check: if Optiland is importable, build an
    equivalent off-axis-conic system there and compare image-plane
    OPD RMS within 1e-4 wave.

    Smoke test only -- the full quantitative cross-check requires
    matching the propagation geometry between Optiland's ray-trace
    OPD and Lumenairy's wave-optics OPD, which is more involved
    than this v5.2 ROADMAP item promises.  We just confirm the
    surface_frame branch runs end-to-end on an Optiland-shaped
    prescription without raising.
    """

    def test_surface_frame_runs_on_off_axis_conic_smoke(self):
        N, dx, wl = 64, 8e-6, 1.55e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        prescription = _flat_singlet_prescription(
            R=15e-3, conic=-0.5,
            decenter=(0.3e-3, 0.1e-3),
            tilt=(2e-3, 1e-3),
            aperture=8e-3)
        E_out = apply_real_lens(
            E_in, prescription=prescription,
            wavelength=wl, dx=dx, bandlimit=False,
            surface_frame=True,
        )
        # Field must be finite over the aperture interior.
        N_aper_radius_px = int(0.5 * 8e-3 / dx)
        i_mid = N // 2
        slc = slice(i_mid - N_aper_radius_px // 2,
                    i_mid + N_aper_radius_px // 2)
        assert np.all(np.isfinite(E_out[slc, slc])), (
            "surface_frame=True produced non-finite values in the "
            "aperture interior on an off-axis-conic prescription."
        )
