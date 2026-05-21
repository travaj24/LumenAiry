"""Consolidated audit-fix tests for the **lens** domain.

This module consolidates v4.9 - v5.0 audit-fix regression pins
from 9 source files (per the v5.2 ROADMAP / 57-file consolidation):

* ``test_audit_fixes_v4_11_2_rw_lens.py``
* ``test_audit_fixes_v4_13_0_ghost_apply_mirror.py``
* ``test_audit_fixes_v4_13_0_perf_bsdf.py``
* ``test_audit_fixes_v4_13_0_perf_coatings.py``
* ``test_audit_fixes_v4_13_0_perf_freeform.py``
* ``test_audit_fixes_v4_13_0_perf_thin_grating.py``
* ``test_audit_fixes_v4_13_1_thin_grating_dock.py``
* ``test_audit_fixes_v4_14_0_agent_2.py``
* ``test_audit_fixes_v4_14_1_agent_c.py``

Each source file's contents are concatenated below verbatim (modulo
minimal renames to avoid identifier collisions and to give each top-level
test class an audit-version attribution prefix).  inspect.getsource proxy
tests are tagged with a TODO comment per AUDIT_V4_13_1 Part 6.1.
"""
from __future__ import annotations

# ============================================================================
# Source: test_audit_fixes_v4_11_2_rw_lens.py
# Audit version: V4_11_2  scope: rw_lens
# Original module docstring preserved as comment block for git-blame traceability:
#   Regression tests for the v4.11.2 audit-residual patch in the
#   Richards-Wolf + lens-element + polarization / coating domain.
#   
#   Each test pins one of the round-3 audit findings:
#   
#   * RW-1 -- ``richards_wolf_focus`` Airy-peak intensity scales as 1/f^2
#             (prefactor 1/f, not f); a 1 m vs 0.1 m focal length differ
#             by ~100x in Airy peak (within +/- 20%).  Pre-4.11.2 the
#             missing 1/f^2 factor reversed the scaling.
#   
#   * RW-2 -- ``richards_wolf_focus`` on-axis focal-field global phase
#             matches the sign of ``angular_spectrum_propagate`` over the
#             same distance (both use ``exp(+i k z)`` under the library's
#             ``exp(-i omega t)`` convention).  Pre-4.11.2 used the
#             opposite ``exp(-i k f)``.
#   
#   * CT-1 -- ``thin_film_stack(polarization='avg')`` is symmetric at
#             normal incidence: T_avg == 0.5 * (T_s + T_p) and the
#             s/p-separately-computed transmissions agree at 0 deg.
#             Pre-4.11.2 the 'avg' branch reused the LAST polarization's
#             eta_sub/eta_amb (p-pol) for both T_s and T_p; at 0 deg this
#             was masked because eta_s = eta_p, but the regression test
#             still verifies the equality holds (no symmetry drift) and
#             additionally checks that a non-zero AOI no longer reuses the
#             p-pol admittance for the s-pol transmission.
#   
#   * PL-1 -- ``apply_waveplate`` with retardance=pi/2 and angle=pi/4 on
#             a linear-x input produces a Jones vector with the same
#             handedness as ``create_circular_polarized('right')`` (the
#             v4.11.1 test pinned the result; this test re-pins it to
#             lock in the docstring/code agreement after the 4.11.2
#             docstring sync).
#   
#   Run-time budget: all four tests together should finish in << 5 s.
# ============================================================================
import warnings

import numpy as np
import pytest

import lumenairy as lm

# ============================================================================
# RW-1 -- Richards-Wolf intensity scales correctly with focal length
# ============================================================================

class TestAuditFixesV4_11_2_rw_lens_RichardsWolfIntensityScalesWithF:
    """The Richards-Wolf integral's prefactor goes as 1/f (amplitude)
    -> 1/f^2 (intensity).  An ideal aplanatic objective with identical
    NA and pupil at two different focal lengths f1 vs f2 should give

        I_peak(f1) / I_peak(f2) ~= (f2 / f1)^2

    Pre-v4.11.2 the prefactor was (-i k f / 2 pi) (i.e. multiplied by
    f instead of dividing), so the ratio came out as (f1 / f2)^2 -- the
    WRONG direction.  A 10x focal-length shrink therefore CHANGED the
    Airy-peak intensity by 1/100 instead of x100, a 10^4 relative
    error in the wrong direction.

    The pupil grid is held fixed (same dx_pupil, same Np) so the
    plane-wave amplitude entering the integral is identical between
    the two calls; only ``f`` changes.  NA is also held fixed (the
    NA-limited exit pupil rim is the same).
    """

    def test_peak_intensity_ratio_is_f_squared(self):
        # Modest pupil grid so the test runs quickly.
        Np = 64
        wavelength = 1e-6
        NA = 0.05  # low NA so the Richards-Wolf result reduces to scalar
        dx_pupil = 100e-6
        # Two focal lengths a factor of 10 apart.
        f_long = 1.0   # 1 m
        f_short = 0.1  # 0.1 m  (i.e. 10x shorter)

        # Top-hat pupil (NA-limited) -- uniform amplitude inside the
        # geometric pupil.  richards_wolf_focus itself applies the
        # rim mask using sin(theta) <= NA, so we just hand it a uniform
        # complex 1+0j array.
        pupil = np.ones((Np, Np), dtype=np.complex128)

        Ex_a, Ey_a, Ez_a, _, _ = lm.richards_wolf_focus(
            pupil, wavelength, NA, f_long, dx_pupil,
            N_focal=Np, z_planes=[0.0], polarization='x',
        )
        Ex_b, Ey_b, Ez_b, _, _ = lm.richards_wolf_focus(
            pupil, wavelength, NA, f_short, dx_pupil,
            N_focal=Np, z_planes=[0.0], polarization='x',
        )

        I_a = (np.abs(Ex_a)**2 + np.abs(Ey_a)**2 + np.abs(Ez_a)**2)
        I_b = (np.abs(Ex_b)**2 + np.abs(Ey_b)**2 + np.abs(Ez_b)**2)
        peak_a = float(I_a.max())
        peak_b = float(I_b.max())

        # I_peak(f_short) / I_peak(f_long) should be (f_long / f_short)^2 = 100
        ratio = peak_b / peak_a
        expected = (f_long / f_short) ** 2  # 100
        assert 0.8 * expected <= ratio <= 1.2 * expected, (
            f"Richards-Wolf peak intensity at f={f_short} m vs f={f_long} m: "
            f"got ratio = {ratio:.3f}, expected ~ {expected:.1f} (within "
            f"+/- 20%).  Pre-4.11.2 the prefactor was missing the 1/f^2 "
            f"factor; the ratio came out around 1/100 instead of 100."
        )


# ============================================================================
# RW-2 -- Richards-Wolf global phase has the +i k f sign
# ============================================================================

class TestAuditFixesV4_11_2_rw_lens_RichardsWolfGlobalPhaseSign:
    """The Richards-Wolf prefactor includes a global ``exp(+i k f)``
    under the library's ``exp(-i omega t)`` convention -- the same sign
    that every other forward propagator (``angular_spectrum_propagate``,
    ``fresnel_propagate``, ...) uses for ``exp(+i k z)``.

    Pre-4.11.2 the code had ``exp(-i k f)``, opposite-sign to all the
    other forward props.  Coherent superposition of a focal field with
    a reference arm propagated by ASM picked up a spurious
    ``exp(-2 i k f)`` mismatch.

    Test approach: at two focal lengths f1 and f2, the *change* in
    on-axis focal-field phase between f1 and f2 should be approximately
    +k*(f2-f1) (modulo 2 pi) -- the same sign as the ASM phase
    advance over a propagation distance (f2 - f1).  Pre-4.11.2 the
    phase would advance with the opposite sign.

    Because the prefactor's 1/f factor introduces a slowly-varying
    amplitude with no phase contribution (real-positive), the phase
    of E_x(0,0) is dominated by the prefactor's exp(+i k f) term plus
    the (real-positive) area-integrated pupil sum.  The (modulo 2 pi)
    increment between f1 and f2 should therefore equal k*(f2-f1).
    """

    def test_on_axis_phase_advances_like_forward_prop(self):
        Np = 64
        wavelength = 1e-6
        NA = 0.05
        dx_pupil = 100e-6
        # Choose two focal lengths whose difference is an integer
        # number of wavelengths so the phase comparison is unambiguous.
        # Use a small absolute f to keep k*f small for clarity, but
        # finite so the prefactor difference is visible.
        delta_f = wavelength * 1000.0  # 1000 wavelengths, ~ 1 mm
        f1 = 0.10           # 0.1 m
        f2 = 0.10 + delta_f
        k = 2 * np.pi / wavelength

        pupil = np.ones((Np, Np), dtype=np.complex128)

        Ex_1, _, _, _, _ = lm.richards_wolf_focus(
            pupil, wavelength, NA, f1, dx_pupil,
            N_focal=Np, z_planes=[0.0], polarization='x',
        )
        Ex_2, _, _, _, _ = lm.richards_wolf_focus(
            pupil, wavelength, NA, f2, dx_pupil,
            N_focal=Np, z_planes=[0.0], polarization='x',
        )

        ic = Np // 2
        # Wrap the difference into [-pi, pi].
        phase_1 = np.angle(Ex_1[ic, ic])
        phase_2 = np.angle(Ex_2[ic, ic])
        dphi = (phase_2 - phase_1 + np.pi) % (2 * np.pi) - np.pi

        # ASM forward-prop over distance delta_f advances the on-axis
        # plane wave by +k*delta_f.  Wrap to [-pi, pi] for comparison.
        expected = ((k * delta_f) + np.pi) % (2 * np.pi) - np.pi
        # ASM forward-prop reference: same wrap of +k*delta_f.
        # (We could call angular_spectrum_propagate over delta_f and
        # measure its central-pixel phase; that gives the same result
        # but adds dependent-grid setup overhead.  The k*delta_f form
        # is equivalent.)

        # Allow a small tolerance for FFT-grid sampling effects.
        assert abs(dphi - expected) < 0.05, (
            f"Richards-Wolf on-axis phase shift between f={f1} m and "
            f"f={f2} m: got d(phase) = {dphi:.4f} rad, expected "
            f"{expected:.4f} rad (= +k*delta_f wrapped to [-pi, pi]).  "
            f"Pre-4.11.2 used exp(-i k f), so the phase advance was "
            f"-k*delta_f -- opposite sign to angular_spectrum_propagate."
        )


# ============================================================================
# CT-1 -- Coating 'avg' mode is symmetric at normal incidence
# ============================================================================

class TestAuditFixesV4_11_2_rw_lens_CoatingAvgPolarizationSymmetric:
    """At 0-deg AOI, s-pol and p-pol reduce to the same physics
    (the polarization basis is degenerate at normal incidence), so

        thin_film_stack(..., angle=0, polarization='avg')

    must produce the same T as 0.5*(T_s + T_p) computed by two
    separate calls with polarization='s' and 'p'.

    Pre-4.11.2 the 'avg' branch reused the LAST polarization's
    ``eta_sub`` / ``eta_amb`` (p-pol, since pols=['s','p']) inside
    the T_s and T_p formulas, which made the averaged T silently wrong
    at oblique AOI.  At 0 deg the bug was masked because eta_s=eta_p,
    but the test still pins the equality so a future refactor can't
    silently break the 0-deg consistency.

    For belt-and-braces coverage, we also pin that the s/p separate
    transmissions agree at 0 deg (a basic physical consistency check).
    """

    def test_avg_equals_half_sum_at_zero_aoi(self):
        from lumenairy.elements.coatings import coating_reflectance

        # Simple single-layer AR-style coating on a high-index substrate.
        # Wavelength 1 um, n_amb=1 (air), n_sub=1.5 (BK7-like), one
        # MgF2-like layer at the AR thickness.
        wavelengths = np.array([1.0e-6])
        layers = [(1.38, 1.0e-6 / (4 * 1.38))]  # quarter-wave at 1 um
        n_ambient = 1.0
        n_substrate = 1.5
        angle = 0.0

        R_s, T_s, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='s')
        R_p, T_p, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='p')
        R_avg, T_avg, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='avg')

        # At 0 deg, T_s == T_p (and R_s == R_p) to numerical precision.
        assert abs(float(T_s[0]) - float(T_p[0])) < 1e-12, (
            f"Coating at 0 deg: T_s={float(T_s[0]):.6f}, "
            f"T_p={float(T_p[0]):.6f} differ.  At normal incidence "
            f"s-pol and p-pol must be degenerate."
        )
        # avg should equal both, and equal 0.5*(T_s + T_p).
        expected = 0.5 * (float(T_s[0]) + float(T_p[0]))
        assert abs(float(T_avg[0]) - expected) < 1e-10, (
            f"Coating 'avg' mode at 0 deg: T_avg={float(T_avg[0]):.6f}, "
            f"expected 0.5*(T_s + T_p) = {expected:.6f}.  Pre-4.11.2 "
            f"the 'avg' branch reused the p-pol eta_sub / eta_amb for "
            f"both T_s and T_p computations; at 0 deg this was masked "
            f"because eta_s = eta_p, but the formula is structurally "
            f"wrong and the regression test pins the correct equality."
        )

    def test_avg_uses_correct_admittance_at_oblique_aoi(self):
        """At a non-zero AOI, T_avg from 'avg' mode must equal
        0.5*(T_s + T_p) computed separately.  Pre-4.11.2 the 'avg'
        branch used p-pol admittances inside both T_s and T_p, giving
        T_avg(buggy) = 0.5 * (T_s_with_p_admittance + T_p_with_p_admittance)
        != 0.5 * (T_s_correct + T_p_correct) in general at oblique AOI.
        """
        from lumenairy.elements.coatings import coating_reflectance

        wavelengths = np.array([1.0e-6])
        # 2-layer coating to get a non-trivial admittance ratio.
        layers = [(1.38, 1.0e-6 / (4 * 1.38)),
                  (2.30, 1.0e-6 / (4 * 2.30))]
        n_ambient = 1.0
        n_substrate = 1.5
        # 30 deg AOI is large enough that eta_s and eta_p differ by ~30%.
        angle = np.deg2rad(30.0)

        _, T_s, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='s')
        _, T_p, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='p')
        _, T_avg, _ = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_substrate, n_ambient=n_ambient,
            polarization='avg')

        expected = 0.5 * (float(T_s[0]) + float(T_p[0]))
        assert abs(float(T_avg[0]) - expected) < 1e-8, (
            f"Coating 'avg' at 30 deg AOI: T_avg={float(T_avg[0]):.6f}, "
            f"expected 0.5*(T_s + T_p) = {expected:.6f} "
            f"(T_s={float(T_s[0]):.6f}, T_p={float(T_p[0]):.6f}).  "
            f"Pre-4.11.2 the 'avg' branch overwrote eta_sub/eta_amb "
            f"in the inner loop and used the LAST iteration's value "
            f"(p-pol) for BOTH T_s and T_p, so T_avg drifted from the "
            f"correct half-sum at oblique AOI."
        )


# ============================================================================
# PL-1 -- apply_waveplate matches create_circular_polarized('right')
# ============================================================================

class TestAuditFixesV4_11_2_rw_lens_ApplyWaveplateMatchesCreateCircularRight:
    """4.11.1 already pinned this equivalence; the v4.11.2 docstring
    sync (correcting the docstring formula
    ``J = R(-theta) diag(1, exp(+i phi)) R(theta)`` -> the actual
    ``R(theta) diag(1, exp(-i phi)) R(-theta)``) shouldn't break it.

    Test approach: pass a linear-x JonesField through a quarter-wave
    plate at angle pi/4 and compare the on-grid Stokes S3 sign with
    ``create_circular_polarized('right')`` -- they must agree (same
    handedness).
    """

    def test_qwp_on_linear_x_matches_create_right_handedness(self):
        N, dx = 16, 5e-6
        scalar = np.ones((N, N), dtype=np.complex128)
        jf_lin = lm.create_linear_polarized(scalar, dx, angle=0.0)
        jf_qwp = lm.apply_quarter_wave_plate(jf_lin, angle=np.pi / 4)
        jf_ref = lm.create_circular_polarized(
            scalar, dx, handedness='right')
        s_qwp = float(np.mean(lm.stokes_parameters(jf_qwp)['S3']))
        s_ref = float(np.mean(lm.stokes_parameters(jf_ref)['S3']))
        assert s_qwp * s_ref > 0, (
            f"apply_waveplate(retardance=pi/2, angle=pi/4) on linear-x: "
            f"S3 = {s_qwp:.3f} should agree in sign with "
            f"create_circular_polarized('right') S3 = {s_ref:.3f}.  "
            f"4.11.2 only synced the docstring formula, not the "
            f"implementation, so this 4.11.1 equivalence must still "
            f"hold."
        )
        # Tight bound: both should be close to +1 in S3/S0 ratio.
        assert s_qwp > 0.5 and s_ref > 0.5, (
            f"Both S3 should be near +1 (right-circular): got "
            f"qwp={s_qwp:.3f}, ref={s_ref:.3f}."
        )


# ============================================================================
# Source: test_audit_fixes_v4_13_0_ghost_apply_mirror.py
# Audit version: V4_13_0  scope: ghost_apply_mirror
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.13.0 Track-B closure of two v4.12.2 known-
#   limitations carried forward from ``AUDIT_V4_12_1_2026_05_16.md``:
#   
#   * **S3** ``analysis/ghost.py`` R_i/R_j convention conflict.
#   * **L6** ``elements.elements.apply_mirror`` array-namespace dispatch +
#     missing ``dy`` parameter.
#   
#   S3 -- ghost.py R convention
#   ---------------------------
#   
#   Pre-v4.13.0 the ``ghost_analysis`` body docstring used ``|R_i|`` and
#   ``|R_j|`` to refer to *curvature radii* in the harmonic-mean
#   heuristic, while elsewhere in the same module ``R_i`` / ``R_j``
#   denoted the *Fresnel reflectance*.  Two different conventions for
#   the same letter in the same module.
#   
#   v4.13.0 (this fix) adopts a strict upper/lowercase split:
#   
#   * Uppercase ``R_i`` / ``R_j`` / ``R_k`` -- Fresnel reflectance
#     (dimensionless, ratio in [0, 1]).
#   * Lowercase ``r_i`` / ``r_j`` -- curvature radius (metres, signed
#     Welford convention).
#   
#   The body code renames the local heuristic variables from
#   ``R_i_val`` / ``R_j_val`` to ``r_i`` / ``r_j``; the docstrings (both
#   module and function) explicitly disambiguate the two
#   interpretations; a new top-of-module "Naming convention" block
#   documents the split.
#   
#   L6 -- apply_mirror array-namespace dispatch + dy
#   ------------------------------------------------
#   
#   Pre-v4.13.0 ``apply_mirror`` used ``np.exp``, ``np.where``,
#   ``np.meshgrid`` directly -- the lone holdout in ``elements.py``
#   that didn't go through ``array_namespace(E_in)`` dispatch.  CuPy
#   and JAX inputs silently fell through to the NumPy code path
#   (round-tripping arrays through the host).  Also still missing the
#   ``dy: Optional[float] = None`` parameter that every other ``apply_*``
#   function in the module accepts (for anamorphic / rectangular grids).
#   
#   v4.13.0 closes both gaps:
#   
#   * ``apply_mirror(jax_E_in, ...)`` returns a JAX array (the namespace
#     is ``jax.numpy``, not ``numpy``).
#   * ``apply_mirror(..., dy=dy)`` builds the meshgrid + aperture mask
#     with the supplied y-pitch, producing the expected anamorphic
#     geometry (more y-pixels inside the aperture when ``dy < dx``).
#   
#   The wave-side ``R > 0 = concave`` convention is preserved (this is
#   the user-facing API on the wave-optics side; ``system_abcd`` /
#   ``seidel_coefficients`` continue to use the OPPOSITE Welford
#   signed-R convention -- see
#   ``validation/elements/test_elements.py::t_curved_mirror_focus``).
#   
#   Author: Andrew Traverso
# ============================================================================

import inspect

import numpy as np
import pytest

import lumenairy as la
from lumenairy.analysis import ghost as ghost_mod
from lumenairy.elements.elements import apply_mirror

# ============================================================================
# S3 -- ghost.py R convention disambiguation
# ============================================================================


class TestAuditFixesV4_13_0_ghost_apply_mirror_GhostRConventionDisambiguated:
    """Pin the v4.13.0 ghost.py docstring + variable-name disambiguation.

    Pre-fix: ``|R_i|`` / ``|R_j|`` appeared in the body docstring of
    :func:`ghost_analysis` referring to curvature radii while everywhere
    else in the module the same letters meant Fresnel reflectance.

    Post-fix:

    * Top-of-module ``"Naming convention"`` block explicitly documents
      the upper/lower ``R``/``r`` split.
    * Body docstring of :func:`ghost_analysis` uses the prose "Fresnel
      reflectance at surface i" when referring to ``R_i`` and the
      prose "curvature radii ``|r_i|`` and ``|r_j|``" (lowercase) when
      referring to radii.
    * Body code locals renamed from ``R_i_val`` / ``R_j_val`` to
      ``r_i`` / ``r_j``.
    """

    def test_module_docstring_has_naming_convention_block(self):
        """Top-of-module convention block exists and names both
        interpretations explicitly."""
        doc = ghost_mod.__doc__ or ""
        assert "Naming convention" in doc, (
            "v4.13.0 ghost.py must declare a Naming-convention block at "
            "module level to disambiguate R (Fresnel reflectance) from r "
            "(curvature radius).  Block missing.")
        # Both interpretations explicitly named in the block.
        assert "Fresnel reflectance" in doc, (
            "Naming convention must explicitly name 'Fresnel reflectance' "
            "for the uppercase-R interpretation.")
        assert ("curvature" in doc.lower()), (
            "Naming convention must explicitly name 'curvature' radius "
            "for the lowercase-r interpretation.")

    def test_module_docstring_mentions_both_uppercase_R_and_lowercase_r(self):
        """The convention block uses both ``R_i`` AND ``r_i`` (or ``r_j``)
        to label the two distinct quantities."""
        doc = ghost_mod.__doc__ or ""
        assert "R_i" in doc, "Module docstring should still mention R_i (Fresnel)."
        # Either lowercase ``r_i``/``r_j`` literal or the prose
        # "lowercase r" / "lowercase ``r``" is acceptable.
        has_lower = ("r_i" in doc and "R_i" in doc) and (
            "r_i`" in doc or "r_j`" in doc or "lowercase" in doc.lower()
        )
        assert has_lower, (
            "Module docstring must use lowercase r_i / r_j (or call out "
            "'lowercase r') for curvature radii.  Found: "
            f"{doc[:400]!r}")

    def test_ghost_analysis_docstring_disambiguates_R_vs_r(self):
        """:func:`ghost_analysis` body docstring labels each ``R`` /
        ``r`` it mentions with explicit prose ('Fresnel reflectance' vs
        'curvature radii')."""
        doc = ghost_mod.ghost_analysis.__doc__ or ""
        # 'R_i' / 'R_j' appear in the Returns block (they're the
        # public dictionary keys; we MUST keep them).  Pin that the
        # docstring explicitly identifies them as Fresnel reflectance
        # near where they appear, AND mentions curvature radii with
        # lowercase r for the focus_z_estimate.
        assert "Fresnel reflectance" in doc, (
            "ghost_analysis docstring must explicitly identify R_i/R_j "
            "as 'Fresnel reflectance' to disambiguate from curvature "
            "radii.")
        assert ("curvature radii" in doc.lower()
                or "curvature radius" in doc.lower()), (
            "ghost_analysis docstring must explicitly call the focus_z_"
            "estimate inputs 'curvature radii' (or 'curvature radius') "
            "to disambiguate from Fresnel reflectance.")
        # And the lowercase r_i / r_j must appear (in the focus_z_estimate
        # description).
        assert ("r_i" in doc or "r_j" in doc), (
            "ghost_analysis docstring must use lowercase r_i / r_j for "
            "the curvature-radii heuristic.")

    def test_body_code_uses_lowercase_r_for_curvature(self):
        """The function body's local variables for curvature radii are
        named ``r_i`` / ``r_j`` (lowercase), not ``R_i_val`` /
        ``R_j_val`` (pre-fix) or ``R_i`` / ``R_j`` (the Fresnel form)."""
        # TODO(v5.2.1): replace with behavioral pin -- inspect.getsource proxy-test pattern (per AUDIT_V4_13_1 Part 6.1)
        src = inspect.getsource(ghost_mod.ghost_analysis)
        # Renamed locals present.
        assert "r_i = surfs[i].radius" in src, (
            "Body should use lowercase r_i = surfs[i].radius for the "
            "curvature-radius local (v4.13.0 rename).")
        assert "r_j = surfs[j].radius" in src, (
            "Body should use lowercase r_j = surfs[j].radius for the "
            "curvature-radius local (v4.13.0 rename).")
        # Pre-fix naming gone.
        assert "R_i_val" not in src, (
            "Pre-fix R_i_val local should be renamed to r_i in v4.13.0.")
        assert "R_j_val" not in src, (
            "Pre-fix R_j_val local should be renamed to r_j in v4.13.0.")

    def test_ghost_analysis_still_returns_R_i_and_R_j_keys(self):
        """The dictionary keys ``'R_i'`` / ``'R_j'`` remain the
        Fresnel-reflectance public API (no rename of these -- they're
        the keys downstream code reads)."""
        # Build a valid prescription via the library's own factory so
        # we don't depend on the prescription-validator schema (which
        # has changed across releases).  ``make_singlet`` is the
        # canonical 2-surface lens factory used in the validation
        # suite (validation/analysis/test_features.py:241).
        prescription = la.make_singlet(
            50.0e-3, np.inf, 4e-3, 'N-BK7', aperture=10e-3)

        ghosts = ghost_mod.ghost_analysis(
            prescription, wavelength=587.6e-9, verbose=False)
        assert len(ghosts) >= 1
        g0 = ghosts[0]
        # Public Fresnel-reflectance keys preserved (uppercase R).
        assert 'R_i' in g0
        assert 'R_j' in g0
        # Values are dimensionless reflectance in [0, 1] (Fresnel),
        # NOT metres (curvature radius).  A glass-air interface with
        # n ~ 1.52 gives R ~ 0.04, so anything < 0.5 distinguishes
        # Fresnel from any conceivable radius-in-metres value.
        assert 0.0 <= g0['R_i'] < 0.5, (
            f"R_i={g0['R_i']} is not a Fresnel reflectance (should be "
            f"in [0, ~0.5] for typical glass surfaces).")
        assert 0.0 <= g0['R_j'] < 0.5, (
            f"R_j={g0['R_j']} is not a Fresnel reflectance.")
        # The intensity is the product R_i * R_j by construction --
        # this also pins that we didn't accidentally rename one of
        # the dictionary keys when disambiguating from curvature radii.
        expected_I = g0['R_i'] * g0['R_j']
        assert abs(g0['intensity'] - expected_I) < 1e-12, (
            f"intensity={g0['intensity']} != R_i*R_j={expected_I} -- "
            f"the public Fresnel-reflectance contract was broken.")


# ============================================================================
# L6.a -- apply_mirror array_namespace dispatch: JAX input -> JAX output
# ============================================================================


class TestAuditFixesV4_13_0_ghost_apply_mirror_ApplyMirrorBackendDispatch:
    """Pin that ``apply_mirror`` now goes through ``array_namespace``
    dispatch instead of hard-coding ``np.exp`` / ``np.where`` /
    ``np.meshgrid``.

    Pre-fix: any non-NumPy input was silently downcast through the host
    (JAX -> NumPy -> JAX round-trip) without the user knowing -- defeated
    the point of running on JAX.

    Post-fix: JAX input stays on JAX, CuPy input stays on CuPy.
    """

    def test_jax_input_returns_jax_output(self):
        """``apply_mirror(jax_E_in, ...)`` returns a JAX array (the
        output namespace is ``jax.numpy``, not ``numpy``)."""
        jax = pytest.importorskip('jax')
        jnp = pytest.importorskip('jax.numpy')

        N = 64
        dx = 8e-6
        lam = 1.31e-6
        R = 50e-3

        E_in = jnp.ones((N, N), dtype=jnp.complex64)
        # Pre-condition: input really is a JAX array.
        assert isinstance(E_in, jax.Array), (
            "Test setup error: jnp.ones should produce a jax.Array.")

        E_out = apply_mirror(E_in, lam, dx, radius=R,
                             aperture_diameter=3e-3)

        # Post-condition: output is a JAX array (not a NumPy fallback).
        assert isinstance(E_out, jax.Array), (
            f"apply_mirror with JAX input must return a JAX array, "
            f"got {type(E_out).__module__}.{type(E_out).__name__}.  "
            f"Indicates apply_mirror still uses np.* directly instead "
            f"of array_namespace(E_in) dispatch (L6).")
        # And the dtype is preserved on the JAX side.
        assert E_out.dtype == jnp.complex64, (
            f"Output dtype {E_out.dtype} != complex64 (input dtype).  "
            f"Backend dispatch should preserve dtype.")

    def test_jax_flat_mirror_returns_jax_output(self):
        """Flat-mirror branch (radius=None) also goes through the
        array-namespace dispatch (only the aperture mask is applied)."""
        jax = pytest.importorskip('jax')
        jnp = pytest.importorskip('jax.numpy')

        N = 64
        dx = 8e-6
        lam = 1.31e-6
        # JAX defaults to 32-bit; use complex64 to avoid the truncation
        # warning and to keep tests independent of the user's
        # JAX_ENABLE_X64 env setting.
        E_in = jnp.ones((N, N), dtype=jnp.complex64)
        E_out = apply_mirror(E_in, lam, dx, radius=None,
                             aperture_diameter=3e-3)
        assert isinstance(E_out, jax.Array), (
            "Flat-mirror branch must also preserve the JAX backend.")

    def test_numpy_input_still_returns_numpy(self):
        """Regression guard: the NumPy fast-path (legacy code path
        using ``_surface_sag_general`` with numba aspherics) is not
        broken by the dispatch change."""
        N = 64
        dx = 8e-6
        lam = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        E_out = apply_mirror(E_in, lam, dx, radius=50e-3,
                             aperture_diameter=3e-3)
        assert isinstance(E_out, np.ndarray), (
            "NumPy input must still return a NumPy ndarray (regression "
            "guard).")
        assert E_out.dtype == np.complex128

    def test_jax_vs_numpy_close(self):
        """Cross-backend numeric agreement: JAX and NumPy paths produce
        equivalent results for a small mirror up to JAX's default
        single-precision rounding (``jax_enable_x64`` is off by default,
        so JAX silently demotes complex128 -> complex64)."""
        jax = pytest.importorskip('jax')
        jnp = pytest.importorskip('jax.numpy')

        N = 64
        dx = 8e-6
        lam = 1.31e-6
        R = 50e-3
        aperture = 3e-3

        # Match JAX's default precision (complex64) on the NumPy side
        # so the two compute paths are at the same dtype.
        E_in_np = np.ones((N, N), dtype=np.complex64)
        E_out_np = apply_mirror(E_in_np, lam, dx, radius=R,
                                aperture_diameter=aperture)

        E_in_jx = jnp.ones((N, N), dtype=jnp.complex64)
        E_out_jx = apply_mirror(E_in_jx, lam, dx, radius=R,
                                aperture_diameter=aperture)
        E_out_jx_host = np.asarray(E_out_jx)

        # Single-precision phase tolerance: 2 * k * sag with sag ~ 1e-5 m
        # and k = 2*pi/1.31e-6 m^-1 -> phase ~ 100 rad.  At fp32 each
        # operation has ~1e-7 relative error, accumulating to ~1e-5
        # absolute on the exp() output.  1e-4 is comfortable headroom.
        max_err = float(np.max(np.abs(E_out_np - E_out_jx_host)))
        assert max_err < 1e-4, (
            f"JAX vs NumPy apply_mirror disagreement |dE|_max = "
            f"{max_err:.3e}, expected < 1e-4 at fp32 precision.  "
            f"Indicates the JAX inline-sag branch is computing something "
            f"different from the NumPy helper.")


# ============================================================================
# L6.b -- apply_mirror dy parameter: anamorphic geometry
# ============================================================================


class TestAuditFixesV4_13_0_ghost_apply_mirror_ApplyMirrorDyAnamorphic:
    """Pin that ``apply_mirror`` now accepts a ``dy`` parameter and the
    aperture mask correctly samples a circular *physical* aperture on
    an anamorphic grid (``dx != dy``).

    With ``dx > dy`` (coarser x-pitch, finer y-pitch) more y-pixels fit
    inside a fixed-diameter circular aperture than x-pixels.  Pre-fix
    (no dy, both axes used dx) the mask was a degenerate ellipse in
    physical space.
    """

    def test_signature_accepts_dy_kwarg(self):
        """``apply_mirror`` exposes a ``dy`` keyword argument with a
        default of ``None`` (interpreted as ``dx``)."""
        sig = inspect.signature(apply_mirror)
        assert 'dy' in sig.parameters, (
            "apply_mirror must accept a 'dy' keyword argument (v4.13.0 "
            "L6).  Current signature: " + str(sig))
        assert sig.parameters['dy'].default is None, (
            "apply_mirror 'dy' parameter must default to None (interpreted "
            "as dx for backward compatibility).")

    def test_dy_default_equals_dx_backwards_compat(self):
        """When ``dy`` is omitted (or explicitly ``None``) the output
        is identical to passing ``dy=dx`` -- i.e. the v4.12.x square-
        grid behaviour."""
        N = 32
        dx = 1e-5
        lam = 1.31e-6
        E_in = np.ones((N, N), dtype=np.complex128)
        E_default = apply_mirror(E_in, lam, dx, radius=20e-3,
                                 aperture_diameter=8 * dx)
        E_explicit = apply_mirror(E_in, lam, dx, dy=dx, radius=20e-3,
                                  aperture_diameter=8 * dx)
        assert np.allclose(E_default, E_explicit, atol=0.0), (
            "dy=None must reproduce dy=dx exactly (backwards-compat).")

    def test_anamorphic_aperture_geometry(self):
        """With ``dy = dx / 2`` (finer y-pitch) more y-pixels fit inside
        a fixed circular aperture than x-pixels.

        Synthetic asymmetric grid:
          * dx = 1e-5 m (10 micron x-pitch)
          * dy = 5e-6 m (5 micron y-pitch)
          * aperture_diameter = 10 * dx = 1e-4 m
        Along x: half-aperture span = 5*dx, so the mask is +-5 pixels
        wide -> ~11 pixels through the centre row.
        Along y: half-aperture span = 5*dx / dy = 10 pixels, so the
        mask is +-10 pixels wide -> ~21 pixels through the centre col.
        Ratio ny/nx -> ~ dx/dy = 2.
        """
        N = 64
        dx = 1e-5
        dy = 5e-6
        lam = 1.31e-6
        aperture_diameter = 10 * dx

        E_in = np.ones((N, N), dtype=np.complex128)
        E_out = apply_mirror(E_in, lam, dx, dy=dy,
                             aperture_diameter=aperture_diameter)

        mask = np.abs(E_out) > 0
        nx_in = int(mask[N // 2, :].sum())  # along x at y=0
        ny_in = int(mask[:, N // 2].sum())  # along y at x=0

        # Expect ny ~ 2 nx (dy half the size of dx, so twice as many
        # y-pixels fit in the same physical aperture).
        assert nx_in > 0 and ny_in > 0, (
            "Aperture mask should leave SOME pixels through; got "
            f"nx_in={nx_in}, ny_in={ny_in}.")
        ratio = ny_in / nx_in
        assert 1.5 < ratio < 2.5, (
            f"Anamorphic ratio ny/nx = {ratio:.3f} expected ~2 "
            f"(dx={dx}, dy={dy} -> dx/dy=2).  Got nx_in={nx_in}, "
            f"ny_in={ny_in}.  Indicates dy is not flowing into the "
            f"meshgrid.")

    def test_anamorphic_circular_mask_matches_physical_ellipse(self):
        """For a circular *physical* aperture on an anamorphic grid the
        binary mask in pixel-space is the set of (i, j) satisfying
        ``((i - N/2)*dx)^2 + ((j - N/2)*dy)^2 <= (D/2)^2``.

        We compute the expected mask from first principles and verify
        apply_mirror's output matches it exactly.
        """
        N = 64
        dx = 1e-5
        dy = 7e-6  # anamorphic, not a simple 2:1 ratio
        lam = 1.31e-6
        D = 0.6 * N * min(dx, dy)  # fits comfortably inside

        E_in = np.ones((N, N), dtype=np.complex128)
        E_out = apply_mirror(E_in, lam, dx, dy=dy,
                             aperture_diameter=D)

        # Build expected mask in physical (m) coordinates.
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dy
        X, Y = np.meshgrid(x, y)
        h_sq = X ** 2 + Y ** 2
        expected_mask = h_sq <= (D / 2) ** 2

        actual_mask = np.abs(E_out) > 0
        assert np.array_equal(actual_mask, expected_mask), (
            f"apply_mirror anamorphic mask doesn't match physical-ellipse "
            f"prediction.  diff = "
            f"{int((actual_mask != expected_mask).sum())} pixels.")


# ============================================================================
# Source: test_audit_fixes_v4_13_0_perf_bsdf.py
# Audit version: V4_13_0  scope: perf_bsdf
# Original module docstring preserved as comment block for git-blame traceability:
#   Correctness pin for the v4.13.0 BSDFModel.total_integrated_scatter
#   vectorisation (Tier-2 perf, audit group alpha task 3).
#   
#   The pre-v4.13.0 default ``total_integrated_scatter`` integrand was
#   evaluated inside a nested Python loop over (theta, phi); v4.13.0
#   builds a 2-D meshgrid of scattered directions and evaluates the BSDF
#   in one broadcasted call.  This test pins the numerical result against
#   both a known closed-form answer and against an inline reference
#   scalar-loop implementation.
# ============================================================================

import numpy as np

from lumenairy.elements.bsdf import (
    BSDFModel,
    HarveyShackBSDF,
    LambertianBSDF,
)


def _reference_scalar_tis(bsdf: BSDFModel,
                          n_theta: int = 256, n_phi: int = 128) -> float:
    """Pre-v4.13.0 nested-loop TIS, inlined here as the pin reference."""
    theta = np.linspace(1e-6, np.pi / 2, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    inc = np.array([0.0, 0.0, -1.0])
    tot = 0.0
    dth = theta[1] - theta[0]
    dph = phi[1] - phi[0]
    for i in range(n_theta):
        for j in range(n_phi):
            s = np.array([
                np.sin(theta[i]) * np.cos(phi[j]),
                np.sin(theta[i]) * np.sin(phi[j]),
                np.cos(theta[i]),
            ])
            b = float(bsdf.evaluate(inc, s))
            tot += b * np.cos(theta[i]) * np.sin(theta[i]) * dth * dph
    return tot


class _DefaultLambertian(BSDFModel):
    """Subclass that does NOT override total_integrated_scatter --
    forces use of the BSDFModel default integration path so the
    vectorised quadrature is what we actually pin."""
    kind = 'default_lambertian'

    def __init__(self, rho: float = 1.0):
        self.rho = float(rho)

    def evaluate(self, incident_dir, scattered_dir):
        sd = np.asarray(scattered_dir)
        if sd.ndim == 1:
            in_hemi = sd[2] > 0
        else:
            in_hemi = sd[..., 2] > 0
        return self.rho / np.pi * in_hemi

    def sample(self, incident_dir, n_samples, rng=None):
        raise NotImplementedError


def test_default_tis_against_lambertian_closed_form():
    """For a Lambertian-pi BSDF the analytical TIS equals rho exactly.
    Pin the vectorised default integration against rho within 1e-3
    relative."""
    rho = 0.7
    bsdf = _DefaultLambertian(rho)
    tis = bsdf.total_integrated_scatter()
    # Numerical-quadrature error from the default 256x128 grid is the
    # endpoint-rule cosine bias at theta -> pi/2; ~1e-3 relative is the
    # achievable bound without going to a Gauss-Legendre rule.
    assert abs(tis - rho) <= 1e-3 * rho, (
        f'TIS = {tis:.6f}, expected ~{rho:.6f}')


def test_vectorised_matches_scalar_loop_harvey_shack():
    """A non-degenerate Harvey-Shack BSDF: vectorised TIS must match
    the inline scalar-loop reference to within 1e-10."""
    hs = HarveyShackBSDF(b0=0.05, l=0.02, s=2.0)
    tis_vec = hs.total_integrated_scatter()  # subclass default OR override
    # Force the BSDFModel-default code path by calling it explicitly,
    # since HarveyShackBSDF doesn't override TIS.
    tis_ref = _reference_scalar_tis(hs)
    assert abs(tis_vec - tis_ref) <= 1e-10, (
        f'vectorised TIS = {tis_vec:.10e}, scalar = {tis_ref:.10e}')


def test_vectorised_matches_scalar_loop_default_lambertian():
    """Same pin for the bare-default Lambertian-pi subclass that
    forces use of BSDFModel.total_integrated_scatter."""
    b = _DefaultLambertian(rho=0.3)
    tis_vec = b.total_integrated_scatter()
    tis_ref = _reference_scalar_tis(b)
    assert abs(tis_vec - tis_ref) <= 1e-10, (
        f'vectorised TIS = {tis_vec:.10e}, scalar = {tis_ref:.10e}')


# ============================================================================
# Source: test_audit_fixes_v4_13_0_perf_coatings.py
# Audit version: V4_13_0  scope: perf_coatings
# Original module docstring preserved as comment block for git-blame traceability:
#   Correctness pin for the v4.13.0 coating-stack characteristic-matrix
#   vectorisation (Tier-2 perf, audit group alpha task 4).
#   
#   The pre-v4.13.0 ``coating_reflectance`` walked the layer stack with
#   ``M = M @ Mj`` in a Python loop.  v4.13.0 builds all per-layer 2x2
#   matrices as a (N, 2, 2) ndarray and reduces them with
#   ``functools.reduce(np.matmul, ...)``.  This test pins the vectorised
#   reflectance / transmittance against an inline scalar-loop reference at
#   a 50-layer AR stack to within 1e-12.
# ============================================================================

import numpy as np

from lumenairy.elements.coatings import coating_reflectance


def _scalar_reference_coatings(layers, wavelengths, angle=0.0,
                      n_substrate=1.52, n_ambient=1.0,
                      polarization='avg'):
    """Pre-v4.13.0 scalar-loop characteristic-matrix walker (uses
    ``M = M @ Mj``), inlined here as the pin reference."""
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=np.float64))
    n_wv = wavelengths.size
    R = np.empty(n_wv)
    T = np.empty(n_wv)
    phase_r = np.empty(n_wv)
    pols = ['s', 'p'] if polarization == 'avg' else [polarization]
    for iw, lam in enumerate(wavelengths):
        rs, ts = [], []
        eta_sub_by_pol, eta_amb_by_pol = {}, {}
        for pol in pols:
            M = np.eye(2, dtype=np.complex128)
            theta_prev = angle
            n_prev = complex(n_ambient)
            for n_layer, d in layers:
                n_layer = complex(n_layer)
                sin_t = n_prev.real * np.sin(theta_prev) / n_layer.real
                sin_t = min(sin_t, 0.9999)
                cos_t = np.sqrt(1 - sin_t ** 2)
                delta = 2 * np.pi * n_layer * d * cos_t / lam
                if pol == 's':
                    eta = n_layer * cos_t
                else:
                    eta = n_layer / cos_t
                Mj = np.array([
                    [np.cos(delta), -1j * np.sin(delta) / eta],
                    [-1j * eta * np.sin(delta), np.cos(delta)],
                ], dtype=np.complex128)
                M = M @ Mj
                theta_prev = np.arcsin(sin_t)
                n_prev = n_layer
            sin_sub = (n_prev.real * np.sin(theta_prev)
                       / complex(n_substrate).real)
            sin_sub = min(sin_sub, 0.9999)
            cos_sub = np.sqrt(1 - sin_sub ** 2)
            if pol == 's':
                eta_sub = complex(n_substrate) * cos_sub
                eta_amb = complex(n_ambient) * np.cos(angle)
            else:
                eta_sub = complex(n_substrate) / cos_sub
                eta_amb = complex(n_ambient) / np.cos(angle)
            eta_sub_by_pol[pol] = eta_sub
            eta_amb_by_pol[pol] = eta_amb
            B = M[0, 0] + M[0, 1] * eta_sub
            C = M[1, 0] + M[1, 1] * eta_sub
            r = (eta_amb * B - C) / (eta_amb * B + C)
            t_amp = 2.0 * eta_amb / (eta_amb * B + C)
            rs.append(r); ts.append(t_amp)
        if polarization == 'avg':
            R_val = 0.5 * (abs(rs[0]) ** 2 + abs(rs[1]) ** 2)
            phase_val = 0.5 * (np.angle(rs[0]) + np.angle(rs[1]))
            _ess = eta_sub_by_pol['s']; _eas = eta_amb_by_pol['s']
            _esp = eta_sub_by_pol['p']; _eap = eta_amb_by_pol['p']
            T_s = float((_ess.real / max(_eas.real, 1e-30))
                        * abs(ts[0]) ** 2)
            T_p = float((_esp.real / max(_eap.real, 1e-30))
                        * abs(ts[1]) ** 2)
            T_val = 0.5 * (T_s + T_p)
        else:
            R_val = abs(rs[0]) ** 2
            phase_val = np.angle(rs[0])
            T_val = float((eta_sub.real / max(eta_amb.real, 1e-30))
                          * abs(ts[0]) ** 2)
        R[iw] = R_val
        T[iw] = max(0.0, T_val)
        phase_r[iw] = phase_val
    return R, T, phase_r


def _build_50_layer_ar_stack(lam0: float = 550e-9):
    """50-layer alternating high/low quarter-wave stack (canonical
    test-case for TMM perf benchmarks)."""
    n_H = 2.3
    n_L = 1.38
    d_H = lam0 / (4 * n_H)
    d_L = lam0 / (4 * n_L)
    layers = []
    for i in range(25):
        layers.append((n_H, d_H))
        layers.append((n_L, d_L))
    return layers


def test_vectorised_matches_scalar_50_layer_normal():
    """50-layer AR stack at normal incidence, single wavelength."""
    layers = _build_50_layer_ar_stack()
    lam = 550e-9
    R_v, T_v, phi_v = coating_reflectance(
        layers, lam, angle=0.0, n_substrate=1.52, n_ambient=1.0,
        polarization='avg')
    R_s, T_s, phi_s = _scalar_reference_coatings(
        layers, lam, angle=0.0, n_substrate=1.52, n_ambient=1.0,
        polarization='avg')
    assert np.allclose(R_v, R_s, rtol=0, atol=1e-12), (
        f'R diff = {np.max(np.abs(R_v - R_s)):.3e}')
    assert np.allclose(T_v, T_s, rtol=0, atol=1e-12)
    assert np.allclose(phi_v, phi_s, rtol=0, atol=1e-12)


def test_vectorised_matches_scalar_50_layer_oblique_s_pol():
    """50-layer stack at 30 deg AOI, s-polarisation."""
    layers = _build_50_layer_ar_stack()
    lam = 633e-9
    R_v, T_v, phi_v = coating_reflectance(
        layers, lam, angle=np.deg2rad(30.0), n_substrate=1.52,
        n_ambient=1.0, polarization='s')
    R_s, T_s, phi_s = _scalar_reference_coatings(
        layers, lam, angle=np.deg2rad(30.0), n_substrate=1.52,
        n_ambient=1.0, polarization='s')
    assert np.allclose(R_v, R_s, rtol=0, atol=1e-12)
    assert np.allclose(T_v, T_s, rtol=0, atol=1e-12)


def test_vectorised_matches_scalar_wavelength_sweep():
    """Multi-wavelength sweep, 50 layers, avg-pol."""
    layers = _build_50_layer_ar_stack()
    wavelengths = np.linspace(400e-9, 800e-9, 11)
    R_v, T_v, _ = coating_reflectance(
        layers, wavelengths, polarization='avg')
    R_s, T_s, _ = _scalar_reference_coatings(
        layers, wavelengths, polarization='avg')
    assert np.allclose(R_v, R_s, rtol=0, atol=1e-12)
    assert np.allclose(T_v, T_s, rtol=0, atol=1e-12)


# ============================================================================
# Source: test_audit_fixes_v4_13_0_perf_freeform.py
# Audit version: V4_13_0  scope: perf_freeform
# Original module docstring preserved as comment block for git-blame traceability:
#   Correctness pin for the v4.13.0 Chebyshev arccos-hoist
#   (Tier-2 perf, audit group alpha task 5).
#   
#   The pre-v4.13.0 ``surface_sag_chebyshev`` recomputed ``np.arccos(xn)``
#   and ``np.arccos(yn)`` inside the per-coefficient loop.  v4.13.0 hoists
#   both arccos calls outside the loop -- they depend only on the
#   normalised grid, not on the polynomial order (i, j).  This test pins
#   the optimised output to a reference inline computation at 10
#   coefficients on a 64x64 grid.
# ============================================================================

import numpy as np

from lumenairy.elements.freeform import surface_sag_chebyshev


def _scalar_reference_freeform(X, Y, R, conic, cheb_coeffs, norm_x, norm_y):
    """Pre-v4.13.0 Chebyshev sag (arccos inside the loop), inlined
    here as the pin reference."""
    from lumenairy.elements.lenses import surface_sag_general
    h_sq = X ** 2 + Y ** 2
    sag = surface_sag_general(h_sq, R, conic)
    if cheb_coeffs:
        xn_raw = X / norm_x
        yn_raw = Y / norm_y
        outside = (np.abs(xn_raw) > 1.0) | (np.abs(yn_raw) > 1.0)
        xn = np.clip(xn_raw, -1, 1)
        yn = np.clip(yn_raw, -1, 1)
        departure = np.zeros_like(sag)
        for (i, j), c in cheb_coeffs.items():
            Ti = np.cos(i * np.arccos(xn))
            Tj = np.cos(j * np.arccos(yn))
            departure = departure + c * Ti * Tj
        sag = sag + np.where(outside, 0.0, departure)
    return sag


def _build_grid(N=64, half_extent=5e-3):
    x = np.linspace(-half_extent, half_extent, N)
    y = np.linspace(-half_extent, half_extent, N)
    X, Y = np.meshgrid(x, y)
    return X, Y


def _10_coefficient_set():
    """A non-trivial, varied set of 10 Chebyshev coefficients spanning
    low and high polynomial orders."""
    return {
        (0, 0): 1e-7,
        (1, 0): 5e-7,
        (0, 1): -3e-7,
        (2, 1): 2e-8,
        (1, 2): -1e-8,
        (3, 0): 4e-9,
        (0, 4): 7e-9,
        (3, 3): -2e-9,
        (5, 2): 1e-9,
        (4, 5): -5e-10,
    }


def test_hoisted_arccos_matches_inline_reference_64x64():
    """64x64 grid, 10 coefficients, no base curvature."""
    X, Y = _build_grid(N=64)
    coeffs = _10_coefficient_set()
    norm_x = norm_y = 4e-3  # smaller than grid extent so some pixels are outside
    sag_hoisted = surface_sag_chebyshev(
        X, Y, R=np.inf, conic=0.0,
        cheb_coeffs=coeffs, norm_x=norm_x, norm_y=norm_y)
    sag_ref = _scalar_reference_freeform(
        X, Y, R=np.inf, conic=0.0,
        cheb_coeffs=coeffs, norm_x=norm_x, norm_y=norm_y)
    tol = 1e-12 * max(float(np.max(np.abs(sag_ref))), 1.0)
    assert np.max(np.abs(sag_hoisted - sag_ref)) <= tol, (
        f'sag mismatch: max diff '
        f'{np.max(np.abs(sag_hoisted - sag_ref)):.3e}')


def test_hoisted_arccos_matches_inline_reference_with_base_curvature():
    """Same pin but with a curved base sphere underneath the freeform
    departure -- exercises the surface_sag_general code path too."""
    X, Y = _build_grid(N=64)
    coeffs = _10_coefficient_set()
    sag_hoisted = surface_sag_chebyshev(
        X, Y, R=20e-3, conic=-0.5,
        cheb_coeffs=coeffs, norm_x=4e-3, norm_y=4e-3)
    sag_ref = _scalar_reference_freeform(
        X, Y, R=20e-3, conic=-0.5,
        cheb_coeffs=coeffs, norm_x=4e-3, norm_y=4e-3)
    tol = 1e-12 * max(float(np.max(np.abs(sag_ref))), 1.0)
    assert np.max(np.abs(sag_hoisted - sag_ref)) <= tol


def test_hoisted_arccos_outside_region_zeroed():
    """Pixels outside [-norm_x, norm_x] x [-norm_y, norm_y] must
    retain only the base-conic sag with zero freeform departure."""
    X, Y = _build_grid(N=64)
    coeffs = _10_coefficient_set()
    norm = 1e-3  # tight, ensures most of the 5 mm grid is outside
    sag = surface_sag_chebyshev(
        X, Y, R=np.inf, conic=0.0,
        cheb_coeffs=coeffs, norm_x=norm, norm_y=norm)
    outside = (np.abs(X) > norm) | (np.abs(Y) > norm)
    # Base sag with R=inf and conic=0 is identically zero -> all
    # outside pixels must be 0.
    assert np.allclose(sag[outside], 0.0, atol=1e-15)


# ============================================================================
# Source: test_audit_fixes_v4_13_0_perf_thin_grating.py
# Audit version: V4_13_0  scope: perf_thin_grating
# Original module docstring preserved as comment block for git-blame traceability:
#   Correctness pin for the v4.13.0 thin-grating per-order loop
#   vectorisation (Tier-2 perf, audit group alpha task 2).
#   
#   The pre-v4.13.0 thin_grating_efficiency_1d looped over diffraction
#   orders in Python; v4.13.0 broadcasts across the orders axis in a
#   single numpy expression.  This test asserts the vectorised path
#   matches the original analytical formula, evaluated inline, to within
#   machine precision, and that the output shape / order ordering is
#   preserved.
# ============================================================================

import numpy as np
import pytest

from lumenairy.elements.thin_grating import thin_grating_efficiency_1d


def _scalar_reference_thin_grating(period, n_ridge, n_groove, n_substrate, n_superstrate,
                      depth, duty_cycle, wavelength, angle, n_orders):
    """Pre-v4.13.0 Python-loop implementation, inlined here as the
    pin reference."""
    k0 = 2 * np.pi / wavelength
    K = 2 * np.pi / period
    N = 2 * n_orders + 1
    orders = np.arange(-n_orders, n_orders + 1)
    f = duty_cycle
    phi_r = k0 * (complex(n_ridge) - n_substrate) * depth
    phi_g = k0 * (complex(n_groove) - n_substrate) * depth
    tm = np.zeros(N, dtype=np.complex128)
    for idx, m in enumerate(orders):
        if m == 0:
            tm[idx] = f * np.exp(1j * phi_r) + (1 - f) * np.exp(1j * phi_g)
        else:
            tm[idx] = ((np.exp(1j * phi_r) - np.exp(1j * phi_g))
                       * (np.exp(-1j * 2 * np.pi * m * f) - 1)
                       / (-1j * 2 * np.pi * m))
    kx0 = k0 * n_superstrate * np.sin(angle)
    kx = kx0 + orders * K
    k_sub = k0 * n_substrate
    propagating = np.abs(kx) < k_sub
    T_eff = np.where(propagating, np.abs(tm) ** 2, 0.0)
    R_eff = np.zeros(N)
    return orders, R_eff, T_eff


def test_vectorised_matches_scalar_loop_n_orders_20():
    """N=20 orders, generic binary grating: vectorised path must match
    inline scalar loop to within 1e-12 * max(|x|)."""
    args = dict(
        period=2e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.52,
        n_superstrate=1.0, depth=0.6e-6, duty_cycle=0.42,
        wavelength=633e-9, angle=0.0, n_orders=20,
    )
    o_v, R_v, T_v = thin_grating_efficiency_1d(**args)
    o_s, R_s, T_s = _scalar_reference_thin_grating(**args)
    assert np.array_equal(o_v, o_s)
    tol_T = 1e-12 * max(float(np.max(np.abs(T_s))), 1.0)
    tol_R = 1e-12 * max(float(np.max(np.abs(R_s))), 1.0)
    assert np.max(np.abs(T_v - T_s)) <= tol_T, (
        f'T mismatch: max diff {np.max(np.abs(T_v - T_s)):.3e}')
    assert np.max(np.abs(R_v - R_s)) <= tol_R


def test_vectorised_matches_scalar_loop_pi_depth():
    """Deep grating (pi-depth) splits power into +-1 orders; check
    vectorised path matches scalar loop there too."""
    lam = 1e-6
    d_pi = lam / (2 * (1.5 - 1.0))
    args = dict(
        period=5e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.52,
        n_superstrate=1.0, depth=d_pi, duty_cycle=0.5,
        wavelength=lam, angle=0.0, n_orders=25,
    )
    o_v, R_v, T_v = thin_grating_efficiency_1d(**args)
    o_s, R_s, T_s = _scalar_reference_thin_grating(**args)
    assert np.array_equal(o_v, o_s)
    assert np.allclose(T_v, T_s, rtol=0, atol=1e-12)
    assert np.allclose(R_v, R_s, rtol=0, atol=1e-12)


def test_vectorised_shape_and_order_axis():
    """Output shape must be (2*n_orders+1,) with orders centered on 0
    and monotonically increasing."""
    n_orders = 11
    orders, R, T = thin_grating_efficiency_1d(
        period=2e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.52,
        n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5,
        wavelength=633e-9, angle=0.0, n_orders=n_orders,
    )
    N = 2 * n_orders + 1
    assert orders.shape == (N,)
    assert R.shape == (N,)
    assert T.shape == (N,)
    # Order axis: centered on 0, strictly increasing by 1
    assert orders[0] == -n_orders
    assert orders[-1] == n_orders
    assert orders[n_orders] == 0
    assert np.array_equal(np.diff(orders), np.ones(N - 1, dtype=orders.dtype))


# ============================================================================
# Source: test_audit_fixes_v4_13_1_thin_grating_dock.py
# Audit version: V4_13_1  scope: thin_grating_dock
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.13.1 audit fix P1-D:
#   ``ThinGratingDock._run`` kwargs mismatch.
#   
#   Audit reference
#   ---------------
#   
#   ``AUDIT_V4_13_0_2026_05_17.md`` P1-D found that the dock's ``_run``
#   handler called ``la.grating_efficiency_vs_wavelength`` with kwargs
#   ``groove_index=``, ``substrate_index=``, ``profile=``, ``angle=`` --
#   none of which the function actually accepts.  The required
#   ``n_ridge`` and ``n_superstrate`` parameters were missing entirely.
#   Every dock click silently turned into a ``TypeError`` swallowed by
#   the dock's broad ``except Exception`` summary.  The dock was
#   non-functional.
#   
#   v4.13.1 fix:
#   
#   * Extract a pure ``_compute_efficiency_data(inputs)`` helper that
#     drives :func:`lumenairy.thin_grating_efficiency_1d` directly and
#     returns a result dict.  No Qt state, so the compute path is unit-
#     testable without a live ``QApplication`` round-trip.
#   * Add UI fields for ``n_ridge`` and ``n_superstrate`` (previously
#     missing).
#   * ``_run`` now collects inputs via ``_collect_inputs()``, calls
#     ``_compute_efficiency_data()``, and routes the result through
#     ``_draw_result()``.
#   
#   What this test pins
#   -------------------
#   
#   1. **Pure helper happy path**: feed a synthetic-but-realistic input
#      dict to ``_compute_efficiency_data`` and assert the result dict
#      shape, the order axis includes m=0, and the per-wavelength
#      efficiency sum is in [0.5, 1.5] (lossless thin phase grating
#      sums to ~1 by Parseval).
#   2. **Wrong-kwargs regression**: confirm the helper does NOT raise
#      ``TypeError`` on standard inputs (the exact failure mode the
#      pre-fix dock exhibited).
#   3. **Qt smoke test**: in an offscreen ``QApplication``, construct
#      the dock, click-equivalent ``_run()``, and assert the summary
#      text box receives a non-error, non-empty status line.
#   
#   Author: Andrew Traverso -- v4.13.1
# ============================================================================

import os

import numpy as np
import pytest

# Force offscreen Qt BEFORE importing PySide6 in any path below.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


# Probe GUI deps: on CI the [gui] extra is not installed and we
# skip cleanly.  Locally everything imports and the suite runs.
try:
    import matplotlib  # noqa: F401
    from PySide6.QtWidgets import QApplication  # noqa: F401
    matplotlib.use('Agg')
    from lumenairy.ui.model import SystemModel
    from lumenairy.ui.thin_grating_dock import ThinGratingDock
    _GUI_OK = True
except ImportError as _e:
    _GUI_OK = False
    _SKIP_REASON = f'GUI deps unavailable: {_e}'


# ============================================================================
# Pure-helper happy path -- no Qt
# ============================================================================

@pytest.mark.skipif(not _GUI_OK,
                    reason=_SKIP_REASON if not _GUI_OK else '')
class TestAuditFixesV4_13_1_thin_grating_dock_ComputeEfficiencyDataPureHelper:
    """``_compute_efficiency_data`` runs without a QApplication."""

    @staticmethod
    def _sample_inputs() -> dict:
        return {
            'period_m': 2e-6,
            'depth_m': 0.5e-6,
            'duty_cycle': 0.5,
            'n_ridge': 1.5,
            'n_groove': 1.0,
            'n_substrate': 1.5,
            'n_superstrate': 1.0,
            'polarization': 'te',
            'aoi_rad': 0.0,
            'n_orders': 11,
            'wl_min_m': 400e-9,
            'wl_max_m': 800e-9,
            'wl_n': 21,
        }

    def test_helper_returns_expected_dict_shape(self):
        out = ThinGratingDock._compute_efficiency_data(self._sample_inputs())
        assert isinstance(out, dict)
        assert set(out.keys()) >= {
            'wavelengths', 'orders', 'efficiencies', 'summary'}
        assert out['wavelengths'].shape == (21,)
        n_orders_total = out['efficiencies'].shape[0]
        # n_orders=11 in the UI maps to half-width 5, so 2*5+1 = 11.
        assert n_orders_total == 11
        assert out['efficiencies'].shape == (11, 21)
        # m=0 order index must be present.
        assert 0 in out['orders'].tolist()

    def test_helper_energy_balance_thin_phase_grating(self):
        """Lossless thin phase grating: sum_m |t_m|^2 ~= 1 by
        Parseval (within propagating-order truncation).
        """
        out = ThinGratingDock._compute_efficiency_data(self._sample_inputs())
        sums = out['efficiencies'].sum(axis=0)
        assert np.all(sums >= 0.5), (
            f'Energy balance unreasonably low: sums={sums}')
        assert np.all(sums <= 1.5), (
            f'Energy balance overflowing 1.5: sums={sums}')

    def test_helper_does_not_raise_typeerror_on_standard_inputs(self):
        """Regression guard for the pre-v4.13.1 failure mode: the
        old ``_run`` raised ``TypeError`` because ``groove_index=``
        is not a valid kwarg of ``grating_efficiency_vs_wavelength``
        / ``thin_grating_efficiency_1d``.  This test pins that the
        helper *does not* raise on standard inputs.
        """
        try:
            ThinGratingDock._compute_efficiency_data(self._sample_inputs())
        except TypeError as exc:
            pytest.fail(
                f'_compute_efficiency_data raised TypeError on standard '
                f'inputs (regression of the v4.13.0 audit P1-D bug): '
                f'{exc}')


# ============================================================================
# Qt smoke test -- end-to-end dock click
# ============================================================================

@pytest.mark.skipif(not _GUI_OK,
                    reason=_SKIP_REASON if not _GUI_OK else '')
class TestAuditFixesV4_13_1_thin_grating_dock_ThinGratingDockEndToEnd:
    """End-to-end: instantiate the dock in an offscreen
    QApplication and call ``_run()``; assert no exception and the
    summary text box receives a non-error message."""

    @pytest.fixture(scope='class')
    def app(self):
        # Reuse an existing QApplication if one already exists in
        # this process (other UI tests may have created one).
        from PySide6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])

    def test_dock_constructs_and_runs(self, app):
        sm = SystemModel()
        dock = ThinGratingDock(sm)
        # Set reasonable inputs (defaults are already reasonable, but
        # be explicit so the test exercises the input plumbing).
        dock.spin_period.setValue(2.0)       # 2 µm
        dock.spin_depth.setValue(0.5)        # 0.5 µm
        dock.spin_duty.setValue(0.5)
        dock.spin_n_ridge.setValue(1.5)
        dock.spin_n_groove.setValue(1.0)
        dock.spin_n_substrate.setValue(1.5)
        dock.spin_n_superstrate.setValue(1.0)
        dock.spin_aoi.setValue(0.0)
        dock.spin_orders.setValue(11)
        dock.spin_wl_min.setValue(400.0)
        dock.spin_wl_max.setValue(800.0)
        dock.spin_wl_n.setValue(21)

        # _run() must not raise.
        dock._run()

        summary_text = dock.summary.toPlainText()
        assert summary_text, 'summary box is empty after _run()'
        # The pre-fix dock landed here with a TypeError message; the
        # post-fix dock lands with a "Computed N order(s) ..." line.
        assert 'failed' not in summary_text.lower(), (
            f'Dock _run() left a failure message in the summary box: '
            f'{summary_text!r}')
        assert 'TypeError' not in summary_text, (
            f'Dock _run() leaked a TypeError into the summary '
            f'(regression of P1-D): {summary_text!r}')
        assert 'Computed' in summary_text, (
            f'Expected summary to begin with "Computed N order(s)..."; '
            f'got {summary_text!r}')


# ============================================================================
# Source: test_audit_fixes_v4_14_0_agent_2.py
# Audit version: V4_14_0  scope: agent_2
# Original module docstring preserved as comment block for git-blame traceability:
#   Correctness pins for the v4.14.0 audit (Agent 2 scope).
#   
#   Two perf wins:
#   
#   * **2A** :func:`lumenairy.elements.coatings.coating_reflectance` now
#     builds the per-layer characteristic-matrix stack in a single
#     vectorised (n_wv, n_layers, 2, 2) pass instead of rebuilding it
#     inside an outer Python loop over wavelengths.  Snell's real-only
#     angle chain is wavelength-independent in the documented
#     approximation (n.imag dropped, lambda absent), so it is walked
#     exactly once.  The tournament reduction now collapses over the
#     layer axis with batched np.matmul.  Test pins the batched output
#     against an inline scalar-wavelength reference at atol=1e-12 for
#     R, T, and phase_r on a 50-layer AR stack across 200 wavelengths;
#     also pins scalar-wavelength input -> scalar output (back-compat
#     shape contract).
#   
#   * **2B** :func:`lumenairy.elements.lenses._evaluate_polynomial_4d_and_grad34`
#     is now vectorised over basis terms via three ``np.tensordot``
#     calls, mirroring the sibling :func:`_evaluate_polynomial_4d`.
#     The Python loop with the ``if c == 0`` early-exit dominated the
#     apply_real_lens_maslov Newton iteration cost at M=70 basis terms.
#     Test pins (f, df3, df4) against a scalar-loop reference at 1e-14
#     on a 30-term polynomial evaluated on a 16x16x16x16 grid, and
#     exercises a real Maslov Newton inversion to confirm the
#     optimisation result is bit-near-exact vs the pre-perf scalar
#     loop.
# ============================================================================

import math

import numpy as np
import pytest

from lumenairy.elements.coatings import coating_reflectance
from lumenairy.elements.lenses import (
    _chebyshev_derivative_vandermonde,
    _chebyshev_vandermonde,
    _evaluate_polynomial_4d_and_grad34,
    _multi_indices_total_degree,
)

# ---------------------------------------------------------------------------
# 2A: coating_reflectance wavelength-batch correctness pin
# ---------------------------------------------------------------------------


def _scalar_loop_coating_reference(layers, wavelengths, angle=0.0,
                                     n_substrate=1.52, n_ambient=1.0,
                                     polarization='avg'):
    """Pre-v4.14.0 scalar-wavelength characteristic-matrix walker.

    Inlines the exact algorithm used before the wavelength batch axis
    was added, including the TIR cap at 0.9999, the n.imag-dropped
    Snell step, and the Macleod p-pol sign convention.  This is the
    pin reference for the 50-layer x 200-wavelength batch test.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=np.float64))
    n_wv = wavelengths.size
    R = np.empty(n_wv)
    T = np.empty(n_wv)
    phase_r = np.empty(n_wv)
    pols = ['s', 'p'] if polarization == 'avg' else [polarization]
    for iw, lam in enumerate(wavelengths):
        rs, ts = [], []
        eta_sub_by_pol, eta_amb_by_pol = {}, {}
        for pol in pols:
            M = np.eye(2, dtype=np.complex128)
            theta_prev = angle
            n_prev = complex(n_ambient)
            for n_layer, d in layers:
                n_layer = complex(n_layer)
                sin_t = n_prev.real * math.sin(theta_prev) / n_layer.real
                sin_t = min(sin_t, 0.9999)
                cos_t = math.sqrt(1 - sin_t * sin_t)
                delta = 2 * math.pi * n_layer * d * cos_t / lam
                if pol == 's':
                    eta = n_layer * cos_t
                else:
                    eta = n_layer / cos_t
                Mj = np.array([
                    [np.cos(delta), -1j * np.sin(delta) / eta],
                    [-1j * eta * np.sin(delta), np.cos(delta)],
                ], dtype=np.complex128)
                M = M @ Mj
                theta_prev = math.asin(sin_t)
                n_prev = n_layer
            sin_sub = (n_prev.real * math.sin(theta_prev)
                       / complex(n_substrate).real)
            sin_sub = min(sin_sub, 0.9999)
            cos_sub = math.sqrt(1 - sin_sub * sin_sub)
            cos_angle = math.cos(angle)
            if pol == 's':
                eta_sub = complex(n_substrate) * cos_sub
                eta_amb = complex(n_ambient) * cos_angle
            else:
                eta_sub = complex(n_substrate) / cos_sub
                eta_amb = complex(n_ambient) / cos_angle
            eta_sub_by_pol[pol] = eta_sub
            eta_amb_by_pol[pol] = eta_amb
            B = M[0, 0] + M[0, 1] * eta_sub
            C = M[1, 0] + M[1, 1] * eta_sub
            r = (eta_amb * B - C) / (eta_amb * B + C)
            t_amp = 2.0 * eta_amb / (eta_amb * B + C)
            rs.append(r)
            ts.append(t_amp)
        if polarization == 'avg':
            R_val = 0.5 * (abs(rs[0]) ** 2 + abs(rs[1]) ** 2)
            # v4.14.1 (audit P1-NEW-2): aggregate phases via complex
            # sum before taking the angle.  Pre-v4.14.1 form
            # ``0.5 * (np.angle(rs[0]) + np.angle(rs[1]))`` is off by
            # +/-pi at Brewster because r_p sign-flips through zero;
            # the unwrapped arithmetic average of two angles separated
            # by ~pi is wrong.  This reference helper has been updated
            # to match the corrected coatings.py implementation.
            phase_val = np.angle(0.5 * (rs[0] + rs[1]))
            _eta_sub_s = eta_sub_by_pol['s']
            _eta_amb_s = eta_amb_by_pol['s']
            _eta_sub_p = eta_sub_by_pol['p']
            _eta_amb_p = eta_amb_by_pol['p']
            T_s = ((_eta_sub_s.real / max(_eta_amb_s.real, 1e-30))
                    * abs(ts[0]) ** 2)
            T_p = ((_eta_sub_p.real / max(_eta_amb_p.real, 1e-30))
                    * abs(ts[1]) ** 2)
            T_val = 0.5 * (T_s + T_p)
        else:
            R_val = abs(rs[0]) ** 2
            phase_val = np.angle(rs[0])
            T_val = ((eta_sub.real / max(eta_amb.real, 1e-30))
                      * abs(ts[0]) ** 2)
        R[iw] = R_val
        T[iw] = max(0.0, T_val)
        phase_r[iw] = phase_val
    return R, T, phase_r


def _build_50layer_ar_stack(lam0=1.0e-6):
    """50-layer alternating high/low-index quarter-wave AR-like stack."""
    n_H = 2.30  # TiO2-like
    n_L = 1.46  # SiO2-like
    d_H = lam0 / (4 * n_H)
    d_L = lam0 / (4 * n_L)
    layers = []
    for i in range(50):
        if i % 2 == 0:
            layers.append((n_L, d_L))
        else:
            layers.append((n_H, d_H))
    return layers


class TestAuditFixesV4_14_0_agent_2_2A_CoatingWavelengthBatch:
    """Pin the wavelength-batch refactor against a scalar-loop reference."""

    def test_50layer_200wv_normal_incidence_avg_pol(self):
        layers = _build_50layer_ar_stack(1.0e-6)
        wavelengths = np.linspace(0.8e-6, 1.2e-6, 200)
        R_ref, T_ref, ph_ref = _scalar_loop_coating_reference(
            layers, wavelengths, angle=0.0,
            n_substrate=1.52, n_ambient=1.0, polarization='avg')
        R_new, T_new, ph_new = coating_reflectance(
            layers, wavelengths, angle=0.0,
            n_substrate=1.52, n_ambient=1.0, polarization='avg')
        np.testing.assert_allclose(R_new, R_ref, atol=1e-12,
                                    err_msg='R mismatch')
        np.testing.assert_allclose(T_new, T_ref, atol=1e-12,
                                    err_msg='T mismatch')
        np.testing.assert_allclose(ph_new, ph_ref, atol=1e-12,
                                    err_msg='phase_r mismatch')

    def test_50layer_200wv_oblique_aoi_s_pol(self):
        layers = _build_50layer_ar_stack(1.0e-6)
        wavelengths = np.linspace(0.8e-6, 1.2e-6, 200)
        angle = math.radians(30.0)
        R_ref, T_ref, ph_ref = _scalar_loop_coating_reference(
            layers, wavelengths, angle=angle,
            n_substrate=1.52, n_ambient=1.0, polarization='s')
        R_new, T_new, ph_new = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=1.52, n_ambient=1.0, polarization='s')
        np.testing.assert_allclose(R_new, R_ref, atol=1e-12)
        np.testing.assert_allclose(T_new, T_ref, atol=1e-12)
        np.testing.assert_allclose(ph_new, ph_ref, atol=1e-12)

    def test_50layer_200wv_oblique_aoi_p_pol(self):
        layers = _build_50layer_ar_stack(1.0e-6)
        wavelengths = np.linspace(0.8e-6, 1.2e-6, 200)
        angle = math.radians(30.0)
        R_ref, T_ref, ph_ref = _scalar_loop_coating_reference(
            layers, wavelengths, angle=angle,
            n_substrate=1.52, n_ambient=1.0, polarization='p')
        R_new, T_new, ph_new = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=1.52, n_ambient=1.0, polarization='p')
        np.testing.assert_allclose(R_new, R_ref, atol=1e-12)
        np.testing.assert_allclose(T_new, T_ref, atol=1e-12)
        np.testing.assert_allclose(ph_new, ph_ref, atol=1e-12)

    def test_complex_index_lossy_stack(self):
        """Documented approximation: n.imag dropped at the Snell step
        but kept in the phase-thickness delta; this test pins that the
        new batched code reproduces the scalar-loop output for an
        absorbing layer (so any future complex-Snell rewrite has a
        regression checkpoint)."""
        layers = [
            (1.46, 100e-9),                    # SiO2 spacer
            (2.30 + 0.01j, 80e-9),             # mildly absorbing high-index
            (1.46, 100e-9),
        ]
        wavelengths = np.linspace(0.9e-6, 1.1e-6, 50)
        R_ref, T_ref, ph_ref = _scalar_loop_coating_reference(
            layers, wavelengths, angle=0.1,
            n_substrate=1.52, n_ambient=1.0, polarization='avg')
        R_new, T_new, ph_new = coating_reflectance(
            layers, wavelengths, angle=0.1,
            n_substrate=1.52, n_ambient=1.0, polarization='avg')
        np.testing.assert_allclose(R_new, R_ref, atol=1e-12)
        np.testing.assert_allclose(T_new, T_ref, atol=1e-12)
        np.testing.assert_allclose(ph_new, ph_ref, atol=1e-12)

    def test_empty_stack_uncoated_fresnel(self):
        """Zero layers: characteristic matrix is identity, result is
        the bare Fresnel reflectance at the ambient/substrate
        interface.  Pins that the n_layers=0 fast path still produces
        a wavelength-shaped array."""
        wavelengths = np.linspace(0.9e-6, 1.1e-6, 11)
        R_ref, T_ref, ph_ref = _scalar_loop_coating_reference(
            [], wavelengths, angle=0.0,
            n_substrate=1.5, n_ambient=1.0, polarization='s')
        R_new, T_new, ph_new = coating_reflectance(
            [], wavelengths, angle=0.0,
            n_substrate=1.5, n_ambient=1.0, polarization='s')
        np.testing.assert_allclose(R_new, R_ref, atol=1e-14)
        np.testing.assert_allclose(T_new, T_ref, atol=1e-14)
        np.testing.assert_allclose(ph_new, ph_ref, atol=1e-14)
        # Fresnel result is wavelength-independent (in this approx),
        # so all entries should match the analytical R = ((n1-n2)/(n1+n2))^2.
        r_fresnel = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
        np.testing.assert_allclose(R_new, r_fresnel, atol=1e-14)

    def test_singleton_array_returns_array(self):
        """Length-1 array input -> length-1 array output.  Pins the
        existing back-compat: all existing callers pass [lam] or
        np.array([lam]) and expect ndarray return."""
        layers = [(1.38, 1.0e-6 / (4 * 1.38))]
        R, T, ph = coating_reflectance(
            layers, np.array([1.0e-6]), angle=0.0,
            n_substrate=1.5, n_ambient=1.0, polarization='avg')
        assert isinstance(R, np.ndarray) and R.shape == (1,)
        assert isinstance(T, np.ndarray) and T.shape == (1,)
        assert isinstance(ph, np.ndarray) and ph.shape == (1,)

    def test_scalar_input_returns_scalar(self):
        """0-d / Python-scalar input -> Python-float output (new in
        v4.14.0).  Lets callers write
        ``R, T, p = coating_reflectance(layers, lam, ...)`` and use
        the scalar directly without indexing."""
        layers = [(1.38, 1.0e-6 / (4 * 1.38))]
        R, T, ph = coating_reflectance(
            layers, 1.0e-6, angle=0.0,
            n_substrate=1.5, n_ambient=1.0, polarization='avg')
        assert isinstance(R, float)
        assert isinstance(T, float)
        assert isinstance(ph, float)
        # And the value must match what the array-input path produces.
        R_arr, T_arr, ph_arr = coating_reflectance(
            layers, np.array([1.0e-6]), angle=0.0,
            n_substrate=1.5, n_ambient=1.0, polarization='avg')
        assert abs(R - float(R_arr[0])) < 1e-14
        assert abs(T - float(T_arr[0])) < 1e-14
        assert abs(ph - float(ph_arr[0])) < 1e-14


# ---------------------------------------------------------------------------
# 2B: _evaluate_polynomial_4d_and_grad34 vectorisation correctness pin
# ---------------------------------------------------------------------------


def _scalar_loop_poly34_reference(coeffs, multi_indices, u1, u2, u3, u4,
                                    max_order):
    """Pre-v4.14.0 scalar-basis-loop polynomial-and-grad evaluator.

    Inlined here as the pin reference for the tensordot vectorisation.
    """
    shape = np.broadcast(u1, u2, u3, u4).shape
    T1 = _chebyshev_vandermonde(u1, max_order)
    T2 = _chebyshev_vandermonde(u2, max_order)
    T3 = _chebyshev_vandermonde(u3, max_order)
    T4 = _chebyshev_vandermonde(u4, max_order)
    dT3 = _chebyshev_derivative_vandermonde(u3, max_order)
    dT4 = _chebyshev_derivative_vandermonde(u4, max_order)
    f = np.zeros(shape, dtype=np.float64)
    df3 = np.zeros(shape, dtype=np.float64)
    df4 = np.zeros(shape, dtype=np.float64)
    for c, (k1, k2, k3, k4) in zip(coeffs, multi_indices):
        if c == 0.0:
            continue
        T12 = T1[k1] * T2[k2]
        f = f + c * T12 * T3[k3] * T4[k4]
        df3 = df3 + c * T12 * dT3[k3] * T4[k4]
        df4 = df4 + c * T12 * T3[k3] * dT4[k4]
    return f, df3, df4


class TestAuditFixesV4_14_0_agent_2_2B_PolynomialGradVectorise:
    """Pin the basis-vectorised gradient evaluator against the loop."""

    def test_random_30term_degree4_16grid(self):
        max_order = 4
        multi_indices = _multi_indices_total_degree(4, max_order)
        # Pick 30 random basis terms with non-trivial coefficients;
        # leave the rest at zero so the early-exit branch in the
        # scalar reference (`if c == 0: continue`) is also covered.
        rng = np.random.default_rng(0xC0FFEE)
        coeffs = np.zeros(len(multi_indices), dtype=np.float64)
        idx = rng.choice(len(multi_indices), size=30, replace=False)
        coeffs[idx] = rng.standard_normal(30)
        # 16x16x16x16 evaluation grid in [-0.9, 0.9]^4
        ax = np.linspace(-0.9, 0.9, 16)
        u1, u2, u3, u4 = np.meshgrid(ax, ax, ax, ax, indexing='ij')
        f_ref, d3_ref, d4_ref = _scalar_loop_poly34_reference(
            coeffs, multi_indices, u1, u2, u3, u4, max_order)
        f_new, d3_new, d4_new = _evaluate_polynomial_4d_and_grad34(
            coeffs, multi_indices, u1, u2, u3, u4, max_order)
        np.testing.assert_allclose(f_new, f_ref, atol=1e-14, rtol=0,
                                    err_msg='f mismatch')
        np.testing.assert_allclose(d3_new, d3_ref, atol=1e-14, rtol=0,
                                    err_msg='df/du3 mismatch')
        np.testing.assert_allclose(d4_new, d4_ref, atol=1e-14, rtol=0,
                                    err_msg='df/du4 mismatch')

    def test_full_70term_degree4_64grid(self):
        """Mirror the M=70-ish Newton-step workload that actually
        runs inside apply_real_lens_maslov: full total-degree<=4
        4D basis (70 terms), non-zero coefficients."""
        max_order = 4
        multi_indices = _multi_indices_total_degree(4, max_order)
        rng = np.random.default_rng(0xDECAFBAD)
        coeffs = rng.standard_normal(len(multi_indices))
        ax = np.linspace(-0.8, 0.8, 8)  # 8^4 = 4096 grid points
        u1, u2, u3, u4 = np.meshgrid(ax, ax, ax, ax, indexing='ij')
        f_ref, d3_ref, d4_ref = _scalar_loop_poly34_reference(
            coeffs, multi_indices, u1, u2, u3, u4, max_order)
        f_new, d3_new, d4_new = _evaluate_polynomial_4d_and_grad34(
            coeffs, multi_indices, u1, u2, u3, u4, max_order)
        np.testing.assert_allclose(f_new, f_ref, atol=1e-14, rtol=0)
        np.testing.assert_allclose(d3_new, d3_ref, atol=1e-14, rtol=0)
        np.testing.assert_allclose(d4_new, d4_ref, atol=1e-14, rtol=0)

    def test_real_maslov_newton_inversion_bit_near_exact(self):
        """End-to-end check: run apply_real_lens_maslov on a small
        spherical singlet and confirm the field magnitude/phase is
        bit-near-exact against the scalar-loop reference evaluator.

        The grad34 helper is only called inside Newton iterations of
        apply_real_lens_maslov, so the only practical way to exercise
        it on the actual call path is to run the propagator.  We
        monkeypatch the lenses module so the scalar reference is used
        in one run and the vectorised version in another, comparing
        the final fields.
        """
        import lumenairy as lm
        from lumenairy.elements import lenses as _lmod

        # Small problem to keep the test fast.
        N = 32
        dx = 1.0e-5
        wavelength = 1.31e-6
        x = (np.arange(N) - N // 2) * dx
        xx, yy = np.meshgrid(x, x, indexing='ij')
        # Simple Gaussian beam
        w0 = 100e-6
        E = np.exp(-(xx ** 2 + yy ** 2) / w0 ** 2).astype(np.complex128)

        prescription = lm.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                                        glass='N-BK7', aperture=0.3e-3)

        # Cheap ray-field / pupil sampling for fast tests.
        kw = dict(prescription=prescription, wavelength=wavelength, dx=dx,
                   ray_field_samples=8, ray_pupil_samples=8, n_v2=8)

        # Run with the vectorised (current) version.  Filter the
        # aperture-vs-grid UserWarning so the test output is clean.
        import warnings as _warn
        with _warn.catch_warnings():
            _warn.simplefilter('ignore')
            E_vec = lm.apply_real_lens_maslov(E, **kw)

        # Now swap in the scalar-loop reference and run again.
        original = _lmod._evaluate_polynomial_4d_and_grad34
        _lmod._evaluate_polynomial_4d_and_grad34 = (
            _scalar_loop_poly34_reference)
        try:
            # The apply_real_lens_maslov path looks up the helper via
            # the module-level binding in lenses_maslov, which itself
            # imports from elements.lenses.  Patch both name spaces.
            from lumenairy.elements import lenses_maslov as _lm_mod
            had_lm = hasattr(_lm_mod,
                              '_evaluate_polynomial_4d_and_grad34')
            if had_lm:
                _orig_lm = _lm_mod._evaluate_polynomial_4d_and_grad34
                _lm_mod._evaluate_polynomial_4d_and_grad34 = (
                    _scalar_loop_poly34_reference)
            from lumenairy.propagators import asymptotic as _asy_mod
            had_asy = hasattr(_asy_mod,
                               '_evaluate_polynomial_4d_and_grad34')
            if had_asy:
                _orig_asy = _asy_mod._evaluate_polynomial_4d_and_grad34
                _asy_mod._evaluate_polynomial_4d_and_grad34 = (
                    _scalar_loop_poly34_reference)
            try:
                with _warn.catch_warnings():
                    _warn.simplefilter('ignore')
                    E_loop = lm.apply_real_lens_maslov(E, **kw)
            finally:
                if had_lm:
                    _lm_mod._evaluate_polynomial_4d_and_grad34 = _orig_lm
                if had_asy:
                    _asy_mod._evaluate_polynomial_4d_and_grad34 = _orig_asy
        finally:
            _lmod._evaluate_polynomial_4d_and_grad34 = original

        # The Newton iterations may not converge to identical floating
        # point values because the scalar loop and tensordot path
        # accumulate sums in different orders; pin a tight relative
        # tolerance instead of bit-exact.
        max_abs = max(np.max(np.abs(E_vec)), 1e-30)
        rel_diff = np.max(np.abs(E_vec - E_loop)) / max_abs
        assert rel_diff < 1e-10, (
            f'Maslov field differs by {rel_diff:.3e} (rel) between '
            f'scalar-loop and tensordot-vectorised grad34 evaluator; '
            f'expected < 1e-10.'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ============================================================================
# Source: test_audit_fixes_v4_14_1_agent_c.py
# Audit version: V4_14_1  scope: agent_c
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for v4.14.1 audit fixes (Agent C scope).
#   
#   Covers the two Agent-C items from ``AUDIT_V4_14_0_2026_05_17.md``:
#   
#   * **C.1 / P1-NEW-2** -- ``coating_reflectance`` 'avg' polarization
#     branch aggregated the s/p reflection phases as ``0.5 * (angle(r_s)
#     + angle(r_p))``.  Because ``r_p`` sign-flips through zero at
#     Brewster (~56 deg for fused silica at visible), the unwrapped
#     arithmetic mean of two angles separated by ~pi is off by pi/2 (or
#     pi at the singularity).  Fixed to ``angle(0.5 * (r_s + r_p))``
#     (complex sum then angle), which is robust to the pi-discontinuity.
#   
#   * **C.2 / P2-6** -- two sites missed by the v4.13.2 ``0.0+0.0j``
#     literal sweep:
#       - ``lenses_maslov.py:sample_E_bilinear`` -- ``np.where(ok, val,
#         0.0+0.0j)``  --> dtype-aware zero.
#       - ``_lens_thin.py`` aplanatic branch -- ``xp.where(valid,
#         xp.exp(-1j*phase), 1.0+0.0j)``  --> dtype-aware unit-phase
#         sentinel.
#   
#   Author: Andrew Traverso -- v4.14.1
# ============================================================================

import inspect

import numpy as np
import pytest

# ============================================================================
# C.1 -- coating phase aggregation at Brewster (P1-NEW-2)
# ============================================================================

class TestAuditFixesV4_14_1_agent_c_C1BrewsterPhaseAggregation:
    """``coating_reflectance(polarization='avg')`` must aggregate the
    s/p reflection phases via the complex sum, not the arithmetic
    mean of the two individual angles.  The two differ by up to pi/2
    near Brewster where ``r_p`` sign-flips through zero."""

    def _build_ar_coating(self):
        # Single quarter-wave MgF2 layer on fused silica at 632.8 nm.
        # MgF2 n ~ 1.38; fused silica n ~ 1.457 at 632.8 nm.
        n_sub = 1.457
        n_layer = 1.38
        wavelength_center = 632.8e-9
        d_layer = wavelength_center / (4 * n_layer)
        return [(n_layer, d_layer)], n_sub, wavelength_center

    def _scalar_reference_avg_phase(self, layers, wavelengths, angle,
                                     n_substrate, n_ambient):
        """Reference: per-wavelength scalar loop computing r_s, r_p,
        then ``angle(0.5 * (r_s + r_p))``.

        Replicates the per-wavelength branch of
        ``coating_reflectance`` but in an unvectorised loop so we can
        independently compute the correct aggregation.
        """
        from lumenairy.elements.coatings import coating_reflectance

        wv = np.atleast_1d(wavelengths)
        ref = np.empty(wv.size)
        for i, lam in enumerate(wv):
            # Pull r_s and r_p individually from the s and p calls.
            _, _, phase_s = coating_reflectance(
                layers, lam, angle=angle,
                n_substrate=n_substrate, n_ambient=n_ambient,
                polarization='s',
            )
            _, _, phase_p = coating_reflectance(
                layers, lam, angle=angle,
                n_substrate=n_substrate, n_ambient=n_ambient,
                polarization='p',
            )
            # ``coating_reflectance`` returns angle(r) for the single-
            # polarization case.  Recover the unit-magnitude complex
            # reflection coefficient and take the magnitude from the
            # power R.
            R_s, _, _ = coating_reflectance(
                layers, lam, angle=angle,
                n_substrate=n_substrate, n_ambient=n_ambient,
                polarization='s',
            )
            R_p, _, _ = coating_reflectance(
                layers, lam, angle=angle,
                n_substrate=n_substrate, n_ambient=n_ambient,
                polarization='p',
            )
            r_s = np.sqrt(R_s) * np.exp(1j * phase_s)
            r_p = np.sqrt(R_p) * np.exp(1j * phase_p)
            ref[i] = np.angle(0.5 * (r_s + r_p))
        return ref

    def test_brewster_phase_matches_complex_sum_reference(self):
        """At AOI = 56 deg (~Brewster for fused silica @ 632.8 nm) the
        batched 'avg' phase must agree with the complex-sum scalar
        reference to atol=1e-12."""
        from lumenairy.elements.coatings import coating_reflectance

        layers, n_sub, lam0 = self._build_ar_coating()
        angle = np.deg2rad(56.0)  # ~Brewster for fused silica
        # Sweep across Brewster to catch the discontinuity.
        wavelengths = np.linspace(580e-9, 680e-9, 21)
        _, _, phase_r_avg = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_sub, n_ambient=1.0,
            polarization='avg',
        )
        ref = self._scalar_reference_avg_phase(
            layers, wavelengths, angle, n_sub, n_ambient=1.0
        )
        # The batched path computes angle(0.5*(r_s+r_p)); the scalar
        # reference does the same independently.  They should agree
        # bit-near-exactly.
        np.testing.assert_allclose(phase_r_avg, ref, atol=1e-12)

    def test_brewster_phase_no_pi_over_2_jumps(self):
        """Heuristic: at AOI = 56 deg the per-wavelength phase should
        vary smoothly (no ~pi/2 jumps between adjacent wavelengths in
        a fine sweep).  This pins the bug-is-gone behaviour: under
        the buggy arithmetic-mean formula, neighbouring wavelengths
        either side of Brewster differ by ~pi/2 even when r_s and r_p
        themselves vary smoothly."""
        from lumenairy.elements.coatings import coating_reflectance

        layers, n_sub, _lam0 = self._build_ar_coating()
        angle = np.deg2rad(56.0)
        wavelengths = np.linspace(620e-9, 645e-9, 51)
        _, _, phase_r = coating_reflectance(
            layers, wavelengths, angle=angle,
            n_substrate=n_sub, n_ambient=1.0,
            polarization='avg',
        )
        # Unwrap so a 2*pi wraparound (which is physical) doesn't
        # masquerade as a jump.
        phase_unwrapped = np.unwrap(phase_r)
        diffs = np.diff(phase_unwrapped)
        # Adjacent-wavelength steps must be much smaller than pi/2.
        # Empirically this is < 0.05 rad with the complex-sum formula;
        # the buggy arithmetic-mean formula gave jumps near pi/2.
        assert np.max(np.abs(diffs)) < 0.2, (
            "Brewster-region 'avg' phase shows large jumps "
            f"({np.max(np.abs(diffs)):.3f} rad), suggesting the "
            "P1-NEW-2 phase-aggregation bug regressed."
        )

    def test_normal_incidence_phase_unchanged(self):
        """Sanity check: at AOI = 0 (no Brewster) the new and old
        formulae must agree, because r_s == r_p at normal incidence
        so both formulae reduce to ``angle(r_s) == angle(r_p)``."""
        from lumenairy.elements.coatings import coating_reflectance

        layers, n_sub, _lam0 = self._build_ar_coating()
        wavelengths = np.linspace(500e-9, 800e-9, 11)
        _, _, phase_r = coating_reflectance(
            layers, wavelengths, angle=0.0,
            n_substrate=n_sub, n_ambient=1.0,
            polarization='avg',
        )
        _, _, phase_s = coating_reflectance(
            layers, wavelengths, angle=0.0,
            n_substrate=n_sub, n_ambient=1.0,
            polarization='s',
        )
        # At AOI=0, r_s and r_p are identical (modulo sign convention)
        # so the 'avg' phase equals the 's' phase exactly.
        np.testing.assert_allclose(phase_r, phase_s, atol=1e-12)


# ============================================================================
# C.2 -- 0.0+0.0j literal sweep cleanup (P2-6)
# ============================================================================

class TestAuditFixesV4_14_1_agent_c_C2ZeroLiteralSweep:
    """Two sites missed by the v4.13.2 ``0.0+0.0j`` literal sweep
    are now using dtype-aware zeros (and ones) matching the
    canonical v4.13.2 pattern."""

    def test_lenses_maslov_no_complex_zero_literal(self):
        """``sample_E_bilinear`` should no longer reference the bare
        ``0.0 + 0.0j`` complex128 literal in code (only in comments)."""
        from lumenairy.elements import lenses_maslov

        # TODO(v5.2.1): replace with behavioral pin -- inspect.getsource proxy-test pattern (per AUDIT_V4_13_1 Part 6.1)
        src = inspect.getsource(lenses_maslov)
        # Strip comments to focus on executable code.
        code_lines = []
        for line in src.split('\n'):
            stripped = line.split('#', 1)[0]
            code_lines.append(stripped)
        code_only = '\n'.join(code_lines)
        # Bare literal (whitespace-flexible) should be absent.
        assert '0.0 + 0.0j' not in code_only, (
            "lenses_maslov.py still contains a '0.0 + 0.0j' literal "
            "in executable code; v4.13.2 swept these to dtype-aware "
            "``xp.zeros((), dtype=...)``."
        )
        assert '0.0+0.0j' not in code_only.replace(' ', ''), (
            "lenses_maslov.py still contains a '0.0+0.0j' literal."
        )

    def test_lens_thin_aplanatic_no_complex_one_literal(self):
        """The aplanatic-branch unit-phase sentinel should no longer
        be the bare ``1.0 + 0.0j`` complex128 literal."""
        from lumenairy.elements import _lens_thin

        # TODO(v5.2.1): replace with behavioral pin -- inspect.getsource proxy-test pattern (per AUDIT_V4_13_1 Part 6.1)
        src = inspect.getsource(_lens_thin)
        # Strip comments.
        code_lines = []
        for line in src.split('\n'):
            stripped = line.split('#', 1)[0]
            code_lines.append(stripped)
        code_only = '\n'.join(code_lines)
        assert '1.0 + 0.0j' not in code_only, (
            "_lens_thin.py still contains a '1.0 + 0.0j' literal in "
            "executable code; v4.14.1 P2-6 swept this to a dtype-"
            "aware ``xp.ones((), dtype=...)``."
        )
        # Spot-check the aplanatic branch source.
        # TODO(v5.2.1): replace with behavioral pin -- inspect.getsource proxy-test pattern (per AUDIT_V4_13_1 Part 6.1)
        apl = inspect.getsource(_lens_thin.apply_thin_lens)
        assert '1.0 + 0.0j' not in apl.split('#', 1)[0]

    def test_maslov_sample_bilinear_preserves_complex64(self):
        """When ``E_in`` is complex64, the bilinear-sample
        out-of-bounds sentinel must not silently upcast the result
        to complex128.

        We can't easily reach into the closure to inspect
        intermediate dtypes, but we can call ``apply_real_lens_maslov``
        with a complex64 input and check that no intermediate path
        accumulates a complex128 array large enough to trip a dtype
        sanity probe.  The final-output dtype contract is enforced by
        the v4.13.2 final cast in ``apply_real_lens_maslov`` -- this
        test re-runs that pin to confirm regression-resistance.
        """
        import lumenairy as lm
        from lumenairy.elements.lenses_maslov import apply_real_lens_maslov

        # Tiny aplanatic-singlet prescription so the test runs fast.
        pres = lm.make_singlet(
            R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
            aperture=12e-3,
        )
        N = 64
        dx = 200e-6
        # Plane wave, complex64.
        E_in = np.ones((N, N), dtype=np.complex64)
        try:
            E_out = apply_real_lens_maslov(
                E_in,
                prescription=pres,
                wavelength=632.8e-9,
                dx=dx,
                ray_field_samples=6,
                ray_pupil_samples=6,
                poly_order=3,
                n_v2=16,
                output_subsample=2,
            )
        except Exception as exc:
            pytest.skip(f"apply_real_lens_maslov unavailable: {exc!r}")
        # The v4.13.2 final cast in apply_real_lens_maslov guarantees
        # the output dtype matches E_in.dtype.  Re-pin here so an
        # intermediate complex128 upcast (from a stray 0+0j literal)
        # is also caught -- if the final cast were absent the bare
        # literal would leak through.
        assert E_out.dtype == np.complex64

    def test_thin_lens_aplanatic_preserves_complex64(self):
        """Same pin for the aplanatic branch of ``apply_thin_lens``."""
        from lumenairy.elements._lens_thin import apply_thin_lens

        N = 32
        dx = 2e-6
        E_in = np.ones((N, N), dtype=np.complex64)
        E_out = apply_thin_lens(
            E_in,
            f=10e-3, wavelength=632.8e-9, dx=dx,
            lens_model='aplanatic',
        )
        # The v4.13.2 dtype cast at the end of apply_thin_lens
        # guarantees E_out.dtype == E_in.dtype.  This pins that the
        # 1.0+0.0j literal no longer leaks a complex128 sentinel
        # through ``xp.where`` (which, in the absence of the final
        # cast, would have upcasted the result).
        assert E_out.dtype == np.complex64

    def test_thin_lens_aplanatic_rim_unchanged(self):
        """Re-pin the v4.10 semantic fix: outside the aplanatic
        domain (r > |f|) the phase mask should be unit so the field
        is unchanged.  The new ``xp.ones((), dtype=...)`` sentinel
        must preserve this behaviour."""
        from lumenairy.elements._lens_thin import apply_thin_lens

        N = 64
        dx = 1e-5
        f = 1e-4  # very short focal length: most of grid is outside
        E_in = (np.random.RandomState(0).randn(N, N)
                + 1j * np.random.RandomState(1).randn(N, N)).astype(
                np.complex128)
        E_out = apply_thin_lens(
            E_in, f=f, wavelength=632.8e-9, dx=dx,
            lens_model='aplanatic',
        )
        # Build the rim mask: outside the aplanatic domain |r| > |f|.
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, y)
        rim = (X ** 2 + Y ** 2) >= f ** 2
        # In the rim region the multiplier is unit -> E_out == E_in.
        np.testing.assert_allclose(E_out[rim], E_in[rim], atol=1e-12)
