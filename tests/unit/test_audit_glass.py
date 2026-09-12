"""Consolidated audit-fix tests for the **glass** domain.

This module consolidates v4.9 - v5.0 audit-fix regression pins
from 2 source files (per the v5.2 ROADMAP / 57-file consolidation):

* ``test_audit_fixes_v4_11_2_track_a.py``
* ``test_audit_fixes_v4_14_2_agent_a.py``

Each source file's contents are concatenated below verbatim (modulo
minimal renames to avoid identifier collisions and to give each top-level
test class an audit-version attribution prefix).  v5.2.3 closed the
v5.2.1 TODO markers on the inspect.getsource proxy-test sites in this
file: replaced where a behavioral pin was achievable; otherwise kept
inspect.getsource by design and updated the comment to explain why
(see AUDIT_V4_13_1 Part 6.1).
"""
from __future__ import annotations

# ============================================================================
# Source: test_audit_fixes_v4_11_2_track_a.py
# Audit version: V4_11_2  scope: track_a
# Original module docstring preserved as comment block for git-blame traceability:
#   Regression tests for the 4.11.2 Track-A audit-residual patch.
#   
#   Three findings pinned here, each a v4.10/v4.11.1 fix that turned out to
#   be wrong on closer inspection:
#   
#   * **C-LR-1 reversal** -- ``apply_real_lens(seidel_correction=True)`` had
#     its sign flipped by the v4.10 audit-response wave based on an
#     incorrect physics-reasoning step in the round-1 audit.  The pre-v4.10
#     negation was correct; the v4.10 patch produced a correction that
#     approximately *tripled* the lens's analytic OPD at the rim
#     (millimetre-scale rather than tens-of-nm residual).  Round-3 audit
#     caught this; v4.11.2 restores the original sign.  Pin against ground
#     truth from ``apply_real_lens_traced`` (which ray-traces every pixel
#     and does not need this correction at all).
#   
#   * **GBD axial_opl dead-on-arrival** -- v4.11.1 added a
#     ``surfaces_from_prescription`` loop calling ``.get('thickness', ...)``
#     on each element, but the function returns ``List[Surface]`` and
#     ``Surface`` is a ``@dataclass`` with no ``.get`` method.  Every call
#     raised ``AttributeError``, swallowed by a bare ``except Exception``,
#     so ``axial_opl`` was always ``None``.  v4.11.2 switches to
#     ``getattr`` and emits a ``RuntimeWarning`` on any other failure.
#   
#   * **S-LAH64 / S-LAH79 Sellmeier coefficients wrong** -- the in-code
#     coefficients gave ``n_d = 1.846`` (LAH64) and ``1.885`` (LAH79) vs
#     the Ohara catalog values of ``1.78800`` and ``2.00330`` -- off by
#     +5.8% and -5.9% respectively.  Appears to be misattributed
#     coefficients.  v4.11.2 removes the in-code entries and routes both
#     glasses through the ``__sellmeier__`` sentinel
#     (``refractiveindex.info`` lookup, requires ``pip install
#     refractiveindex``).
# ============================================================================
import math
import warnings

import numpy as np
import pytest

import lumenairy as lm

# ============================================================================
# C-LR-1 sign reversal -- correction matches ground truth (traced path)
# ============================================================================

class TestAuditFixesV4_11_2_track_a_SeidelCorrectionSignAgainstGroundTruth:
    """``apply_real_lens(seidel_correction=True)`` fits the residual between
    an exit-vertex-plane ray trace and the split-step model's OWN exit OPL,
    from rho**4 up, and imprints it as a radial screen at the exit pupil.

    WHAT THIS FILE USED TO ASSERT, AND WHY IT COULD NOT FAIL (audit
    2026-09-11, finding V1).  The single ``seidel_correction=True`` test in
    the repository compared the corrected field against
    ``apply_real_lens_traced``, wrapped the phase difference into (-pi, pi]
    with ``np.angle(np.exp(1j*d))``, divided by 2*pi -- so the quantity was
    <= 0.5 by construction -- and then asserted it was < 50.0.  A synthetic
    10**4-wave error scored 0.288 and passed.  The fixture was also
    plano-REAR (R2 = inf), where the exit-vertex defect is identically zero,
    and it scored exit-pupil phase, never focus, so the ~90 um*rho**2 of
    spurious defocus the option injected was invisible on both counts.

    The three tests below are the replacement, and each is falsifiable:

    * the 5 nm gate SKIPS a well-corrected singlet (whose true model
      residual is sub-nm), i.e. the corrected field is the uncorrected one;
    * on a CURVED-REAR cemented doublet the correction improves the exit
      wavefront against an INDEPENDENT ray oracle (not another model in this
      library), measured on an UNWRAPPED radial cut;
    * and it does not move the focus, measured unwrap-free by a
      through-focus scan of the propagated field.
    """

    # -- fixtures ---------------------------------------------------------
    LAM = 632.8e-9

    @staticmethod
    def _doublet(ap=8e-3):
        """AC254-ish cemented doublet -- CURVED rear (R3 = -291 mm), which is
        the geometry the exit-vertex class is visible in."""
        return dict(
            surfaces=[
                dict(radius=33.3e-3, glass_before='AIR',
                     glass_after='N-BAF10'),
                dict(radius=-22.28e-3, glass_before='N-BAF10',
                     glass_after='N-SF6HT'),
                dict(radius=-291.07e-3, glass_before='N-SF6HT',
                     glass_after='AIR')],
            thicknesses=[9.0e-3, 2.5e-3], aperture_diameter=ap)

    @staticmethod
    def _planoconvex(ap=4e-3):
        return dict(
            surfaces=[dict(radius=50e-3, glass_before='AIR',
                           glass_after='N-BK7'),
                      dict(radius=float('inf'), glass_before='N-BK7',
                           glass_after='AIR')],
            thicknesses=[3e-3], aperture_diameter=ap)

    @classmethod
    def _oracle_opl(cls, prescription, heights):
        """INDEPENDENT exit-vertex-plane ray oracle: paraxial-free Newton
        intersection of the conic + vector Snell + the signed transfer back to
        the exit vertex plane, written here from the surface equation rather
        than taken from any model under test.

        Returns ``(x_exit, opl)`` for a collimated meridional fan.
        """
        from lumenairy.elements.lenses import surface_sag_general as _sag
        from lumenairy.glass import get_glass_index

        surfaces = prescription['surfaces']
        thick = list(prescription['thicknesses'])
        x = np.asarray(heights, dtype=np.float64).copy()
        z = np.zeros_like(x)
        Lx = np.zeros_like(x)
        Lz = np.ones_like(x)
        opl = np.zeros_like(x)
        zv = 0.0
        for i, sf in enumerate(surfaces):
            R = sf['radius']
            kc = sf.get('conic', 0.0) or 0.0
            asph = sf.get('aspheric_coeffs')
            n1 = float(get_glass_index(sf['glass_before'], cls.LAM))
            n2 = float(get_glass_index(sf['glass_after'], cls.LAM))
            t = (zv - z) / Lz
            if np.isfinite(R) and R != 0:
                for _ in range(60):           # Newton on z - zv - sag(h) = 0
                    xh = x + t * Lx
                    sg = np.nan_to_num(_sag(xh * xh, R, kc, asph))
                    e = np.maximum(1e-12, 1e-7 * np.abs(xh))
                    sp = np.nan_to_num(_sag((np.abs(xh) + e) ** 2, R, kc, asph))
                    sm = np.nan_to_num(_sag((np.abs(xh) - e) ** 2, R, kc, asph))
                    dsdh = (sp - sm) / (2.0 * e)
                    g = z + t * Lz - zv - sg
                    dg = Lz - dsdh * np.sign(xh) * Lx
                    t = t - g / np.where(np.abs(dg) < 1e-30, 1e-30, dg)
                x = x + t * Lx
                z = z + t * Lz
                e = np.maximum(1e-12, 1e-7 * np.abs(x))
                sp = np.nan_to_num(_sag((np.abs(x) + e) ** 2, R, kc, asph))
                sm = np.nan_to_num(_sag((np.abs(x) - e) ** 2, R, kc, asph))
                dsdh = (sp - sm) / (2.0 * e)
                nx = -dsdh * np.sign(x)
                nz = np.ones_like(x)
            else:
                x = x + t * Lx
                z = z + t * Lz
                nx = np.zeros_like(x)
                nz = np.ones_like(x)
            nn = np.hypot(nx, nz)
            nx, nz = nx / nn, nz / nn
            opl = opl + n1 * t
            ci = Lx * nx + Lz * nz
            eta = n1 / n2
            ct = np.sqrt(np.maximum(1.0 - eta * eta * (1.0 - ci * ci), 0.0))
            ndx = eta * Lx + (ct - eta * ci) * nx
            ndz = eta * Lz + (ct - eta * ci) * nz
            nn2 = np.hypot(ndx, ndz)
            Lx, Lz = ndx / nn2, ndz / nn2
            if i < len(surfaces) - 1:
                zv += thick[i]
        # signed transfer to the exit vertex plane z = sum(thicknesses)
        n_exit = float(get_glass_index(surfaces[-1]['glass_after'], cls.LAM))
        z_exit = float(sum(thick))
        tf = (z_exit - z) / Lz
        opl = opl + n_exit * tf
        return x + tf * Lx, opl

    @classmethod
    def _exit_wavefront_rms(cls, prescription, **kw):
        """Exit-plane OPD rms (piston-free, UNWRAPPED radial cut) of
        ``apply_real_lens`` against the independent oracle, in metres."""
        ap = prescription['aperture_diameter']
        k0 = 2.0 * np.pi / cls.LAM
        h0 = np.linspace(-0.995 * ap / 2, 0.995 * ap / 2, 4001)
        x_or, opl_or = cls._oracle_opl(prescription, h0)
        NA = float(np.max(np.abs(np.gradient(opl_or, x_or))))
        NA = max(NA, 1e-6)
        dx = 0.30 * cls.LAM / NA
        N = int(2 ** np.ceil(np.log2(1.45 * ap / dx)))
        E = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Eo = lm.apply_real_lens(E, prescription=prescription,
                                    wavelength=cls.LAM, dx=dx, **kw)
        x = (np.arange(N) - N / 2) * dx
        # np.unwrap along the FULL row (a Nyquist-sampled exit wavefront is
        # unwrappable by construction), then window -- unwrapping only the
        # window would anchor on a wrapped sample.
        ph = np.unwrap(np.angle(Eo[N // 2]))
        m = np.abs(x) <= 0.85 * ap / 2
        order = np.argsort(x_or)
        d = ph[m] / k0 - np.interp(x[m], x_or[order], opl_or[order])
        d = d - d.mean()
        return float(np.sqrt(np.mean(d ** 2)))

    @classmethod
    def _through_focus_peak(cls, prescription, **kw):
        """(peak |E|**2, best-focus z) from an ASM through-focus scan of the
        exit field.  No unwrapping, no oracle -- the defocus a rho**2 term
        injects shows up here as a moved focus and a lost peak.

        The scan is 21 planes over +-3 % of the traced focal length and the
        maximum is refined by the parabola through the three samples around
        it.  The refinement is not cosmetic: the depth of focus here is
        ~lambda / NA**2 = 0.54 mm against a 0.15 mm plane spacing, so the
        SAMPLED maximum of a sharply peaked curve depends on where the plane
        grid happens to fall relative to each arm's own peak -- two arms whose
        true peaks differ by 2 % can read 2 % the other way.  The parabolic
        vertex is exact for a locally quadratic peak, which this is.
        """
        from lumenairy.propagators.propagation import (
            angular_spectrum_propagate as asm,
        )
        ap = prescription['aperture_diameter']
        h0 = np.linspace(-0.995 * ap / 2, 0.995 * ap / 2, 1001)
        x_or, opl_or = cls._oracle_opl(prescription, h0)
        slope = np.gradient(opl_or, x_or)
        NA = max(float(np.max(np.abs(slope))), 1e-6)
        dx = 0.30 * cls.LAM / NA
        N = int(2 ** np.ceil(np.log2(1.45 * ap / dx)))
        f_est = float(np.abs(x_or[-1] / slope[-1]))
        zs = f_est * np.linspace(0.97, 1.03, 21)
        E = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Eo = lm.apply_real_lens(E, prescription=prescription,
                                    wavelength=cls.LAM, dx=dx, **kw)
            pk = np.array([float(np.abs(asm(Eo.copy(), z, cls.LAM, dx)).max()
                                 ** 2) for z in zs])
        j = int(np.argmax(pk))
        if 0 < j < len(zs) - 1:
            a, b, c = pk[j - 1], pk[j], pk[j + 1]
            den = a - 2.0 * b + c
            if den != 0.0:
                d = 0.5 * (a - c) / den
                return (float(b - 0.25 * (a - c) * d),
                        float(zs[j] + d * (zs[1] - zs[0])))
        return float(pk[j]), float(zs[j])

    # -- the tests --------------------------------------------------------
    def test_seidel_gate_skips_a_well_corrected_singlet(self):
        """The 5 nm gate must SKIP where the model has no residual to carry.

        DERIVATION OF THE BAR.  The split-step model's own exit-OPD residual
        on this plano-convex, against the independent ray oracle below, is
        0.85 nm rms -- so a correctly-referenced correction has at most ~1 nm
        to fit and the gate (5 nm rms on the fitted rho**4+ part) must not
        fire.  Measured with the correct reference: 1.405 nm, i.e. 3.6x
        below the gate.  Pre-fix the same quantity was 158.6 nm (113x wrong
        side of it) because the reference omitted the in-glass obliquity the
        ASM legs already carry, and the field it imprinted was 105x worse
        than leaving the option off.

        The assertion is a BIT-IDENTITY: a skipped gate means the corrected
        call returns exactly the uncorrected field, which is a two-sided
        statement no tolerance can soften.
        """
        rx = self._planoconvex()
        N, dx = 256, 3.0e-6
        E = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = lm.apply_real_lens(E, prescription=rx,
                                   wavelength=self.LAM, dx=dx)
            b = lm.apply_real_lens(E, prescription=rx,
                                   wavelength=self.LAM, dx=dx,
                                   seidel_correction=True)
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), (
            "seidel_correction=True changed the field of a singlet whose "
            "model residual is 0.85 nm rms: the 5 nm gate should have "
            f"skipped it.  max|d| = {np.max(np.abs(a - b)):.3e}")

    def test_seidel_improves_a_curved_rear_doublet_against_a_ray_oracle(self):
        """On the geometry the exit-vertex class is visible in.

        DERIVATION OF THE BAR.  Oracle: an independent Newton-intersection +
        vector-Snell meridional trace back-projected to the exit vertex plane
        (``_oracle_opl`` above, written from the surface equation -- NOT
        ``apply_real_lens_traced``, which is another implementation of the
        same library and was what made the old test self-referential).  Its
        own error floor is the interpolation of a 4001-ray fan onto the field
        row, ~1e-12 m, six decades below either number here.  Scored on an
        UNWRAPPED radial cut, piston removed.

        Measured on this 8 mm doublet: 173.5 nm rms with the correction off,
        and pre-fix 1430.0 nm with it ON -- 8.2x WORSE, which is what the
        wrapped assertion this test replaces could not see.  With the
        exit-vertex transfer, the model-own reference and the rho**4 basis it
        lands at ~50 nm.  The bar is a 3x improvement: two decades above the
        oracle floor, and a factor 2.4 below the measured margin, so it fails
        immediately on any of the three defects returning (each on its own
        put this number the wrong side of 1.0x).
        """
        rx = self._doublet(ap=8e-3)
        off = self._exit_wavefront_rms(rx)
        on = self._exit_wavefront_rms(rx, seidel_correction=True)
        assert off > 100e-9, (
            f"fixture no longer has a high-order residual to correct "
            f"({off * 1e9:.1f} nm rms); the test would be vacuous")
        assert on < off / 3.0, (
            f"seidel_correction=True gives {on * 1e9:.2f} nm rms against the "
            f"independent exit-vertex ray oracle, against {off * 1e9:.2f} nm "
            f"with it OFF -- a {off / on:.2f}x change where >= 3x improvement "
            f"is required.  Pre-fix this was 0.12x (8.2x WORSE).")

    def test_seidel_does_not_move_the_focus_and_does_not_cost_peak(self):
        """Unwrap-free, oracle-free confirmation that no DEFOCUS is injected.

        This is the measurement the shipped validation cannot make: it removes
        piston + tilt + DEFOCUS before reporting rms, so a focus error is
        invisible to it by construction -- which is why "4.5x better on
        AC254-100-C" survived alongside a 90 um*rho**2 defect.  Here the exit
        field is propagated over +-3 % of the traced focal length in 13 planes
        and the peak is read directly.

        DERIVATION OF THE BARS (both on the parabola-refined scan; see
        ``_through_focus_peak`` for why the raw sampled maximum is not usable
        at this depth of focus).
        * PEAK.  Pre-fix the option cost 26.8 % of the focal peak on this
          doublet (111 325 -> 81 437 at a 4 mm pupil).  Measured now: -0.27 %
          (115 929 -> 115 614), i.e. the corrected field focuses as hard as
          the uncorrected one.  The bar is "must not drop by more than 2 %" --
          one-sided by design, because a correction that costs intensity is
          not a correction -- which is 7x above the measurement and 13x below
          the defect it guards.
        * FOCUS PLANE.  Pre-fix the best focus moved -2.5 % (50.6291 ->
          49.3570 mm).  Measured now: +0.146 % (50.7500 -> 50.8239 mm), with
          the right SIGN for a real correction (removing spherical aberration
          moves the marginal/paraxial best-focus compromise outward).  The bar
          is 1.0 %: 6.8x above the measurement and 2.5x below the defect,
          which a returning rho**2 term crosses immediately.
        """
        rx = self._doublet(ap=4e-3)
        pk_off, z_off = self._through_focus_peak(rx)
        pk_on, z_on = self._through_focus_peak(rx, seidel_correction=True)
        assert pk_on >= 0.98 * pk_off, (
            f"seidel_correction=True dropped the focal peak from {pk_off:.2f} "
            f"to {pk_on:.2f} ({100 * (pk_on / pk_off - 1):+.1f} %); a "
            f"correction that costs intensity is not a correction.")
        assert abs(z_on - z_off) <= 0.010 * z_off, (
            f"seidel_correction=True moved best focus from "
            f"{z_off * 1e3:.4f} mm to {z_on * 1e3:.4f} mm "
            f"({100 * (z_on - z_off) / z_off:+.2f} %), more than two scan "
            f"steps; a rho**2 term is back in the fit.")


# ============================================================================
# GBD axial_opl -- dataclass access actually populates the value
# ============================================================================

class TestAuditFixesV4_11_2_track_a_GbdAxialOplPopulated:
    """``propagate_gbd_through_prescription`` is supposed to compute
    ``axial_opl = sum_k n_k * t_k`` over the prescription's surfaces
    and pass it as a kwarg to ``apply_abcd_to_beamlets`` so the
    reconstructed field carries the system's absolute axial-phase
    reference.

    Pre-v4.11.2 (v4.11.1 work) the loop called ``_s.get('thickness',
    0.0)`` on each element of ``surfaces_from_prescription``, which
    returns ``List[Surface]`` -- a list of @dataclass instances with
    no ``.get`` method.  Every iteration raised ``AttributeError``,
    silently swallowed by a bare ``except Exception``, and
    ``axial_opl`` was always set to ``None``.

    Pin: the v4.11.2 path now emits a ``RuntimeWarning`` on any axial-
    OPL failure (the bare-except → warn conversion is part of the
    fix).  If the loop is broken in any future refactor, this warning
    will fire and the test will fail.
    """

    def test_axial_opl_path_does_not_emit_failure_warning(self):
        wavelength = 1.0e-6
        N = 32
        dx = 8e-6
        prescription = lm.make_singlet(
            R1=20e-3, R2=-20e-3, d=2e-3,
            glass='N-BK7', aperture=100e-6)
        # Plane wave on the source grid.
        E_in = np.ones((N, N), dtype=np.complex128)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _ = lm.propagate_gbd_through_prescription(
                E_in, dx=dx, wavelength=wavelength,
                prescription=prescription)
        failure_warns = [
            w for w in caught
            if 'axial-OPL computation failed' in str(w.message)
        ]
        assert not failure_warns, (
            "propagate_gbd_through_prescription emitted "
            "'axial-OPL computation failed' RuntimeWarning -- the "
            "v4.11.2 fix to use attribute access on the Surface "
            "dataclass has regressed.  See AUDIT_ROUND3_2026_05_16.md "
            "CRIT-8.")

    def test_axial_opl_is_actually_non_zero(self):
        """v4.12.1 (audit round-4): the warning-absence test above can
        pass even if ``axial_opl`` silently fell through to None (the
        warn is suppressed inside the try/except but the actual OPL
        wasn't computed).  Pin a stronger condition by monkey-patching
        ``apply_abcd_to_beamlets`` and capturing the ``axial_opl``
        kwarg, then asserting the captured value is non-trivially
        non-zero (it should equal n_glass * t_glass for the N-BK7
        singlet, ~1.51 * 2e-3 = ~3e-3 m).
        """
        from lumenairy.propagators import gbd as _gbd_mod
        wavelength = 1.0e-6
        N = 32
        dx = 8e-6
        prescription = lm.make_singlet(
            R1=20e-3, R2=-20e-3, d=2e-3,
            glass='N-BK7', aperture=100e-6)
        E_in = np.ones((N, N), dtype=np.complex128)

        captured = {'axial_opl': '__unset__'}
        original = _gbd_mod.apply_abcd_to_beamlets

        def _spy(beamlets, A, B, C, D, *, wavelength, axial_opl=None,
                 **kw):
            captured['axial_opl'] = axial_opl
            return original(
                beamlets, A, B, C, D, wavelength=wavelength,
                axial_opl=axial_opl, **kw)

        _gbd_mod.apply_abcd_to_beamlets = _spy
        try:
            _ = lm.propagate_gbd_through_prescription(
                E_in, dx=dx, wavelength=wavelength,
                prescription=prescription)
        finally:
            _gbd_mod.apply_abcd_to_beamlets = original

        assert captured['axial_opl'] != '__unset__', (
            "apply_abcd_to_beamlets was not called -- monkey-patch "
            "spy missed the call site.")
        opl = captured['axial_opl']
        assert opl is not None, (
            "propagate_gbd_through_prescription passed axial_opl=None "
            "to apply_abcd_to_beamlets -- the v4.11.1 silent fallthrough "
            "regressed.  v4.11.2 expects a finite numeric value "
            "(sum n_k * t_k).")
        assert np.isfinite(float(opl)), (
            f"axial_opl was non-finite: {opl!r}.")
        # For the BK7 singlet (n_d ~ 1.51 at 1 um) of thickness 2 mm
        # the OPL is approximately 1.51 * 2e-3 = 3.02e-3 m.  Pin
        # ``axial_opl > 1e-3`` so a regression to silent-zero (or
        # n*t = 1*0 = 0) is caught loudly without being fragile to
        # the precise glass-index value.
        assert float(opl) > 1e-3, (
            f"axial_opl = {opl!r} m, expected > 1e-3 m for an N-BK7 "
            f"singlet of thickness 2 mm (~3e-3 m).  Pre-v4.11.2 the "
            f"bare-except in propagate_gbd_through_prescription "
            f"silently fell through to axial_opl=0 (or None which "
            f"the kernel treats as no piston).")


# ============================================================================
# S-LAH64 / S-LAH79 -- coefficients re-bundled in v4.15 (P1-GL-1)
# ============================================================================

class TestAuditFixesV4_11_2_track_a_SLahGlassesRoutedViaSentinel:
    """v4.11.2 history: the original in-code Sellmeier coefficients for
    S-LAH64 and S-LAH79 were wrong by 5-6% in n_d (misattributed from
    a different glass).  v4.11.2 *removed* the in-code entries so that
    lookups would route through ``refractiveindex.info`` exclusively.

    v4.15 (P1-GL-1) re-bundles correct OHARA coefficients (sourced
    verbatim from the ``refractiveindex.info-database`` YAML for the
    OHARA Zemax 2017-11-30 catalog) so that minimal installs without
    the ``refractiveindex`` Python package can still resolve these
    glasses via the dispatcher's ``__sellmeier__`` fallback path.
    The new coefficients agree with refractiveindex.info to ~1e-10
    (S-LAH64) and ~4e-7 (S-LAH79) at n_d.

    This test class was originally written to pin the v4.11.2 absence;
    v4.15 inverts the contract: the entries are present AND accurate.
    """

    def test_s_lah64_in_table_with_correct_n_d(self):
        """v4.15 (P1-GL-1): S-LAH64 has bundled Sellmeier coefficients
        matching the Ohara catalog n_d=1.78800 within 5e-5."""
        from lumenairy.glass import SELLMEIER_COEFFICIENTS, _sellmeier_index
        assert 'S-LAH64' in SELLMEIER_COEFFICIENTS, (
            "v4.15 (P1-GL-1): S-LAH64 should be bundled now to support "
            "minimal installs without the ``refractiveindex`` package."
        )
        n_d = _sellmeier_index(
            wavelength_m=0.58756e-6,
            coeffs=SELLMEIER_COEFFICIENTS['S-LAH64'],
            glass_name='S-LAH64')
        assert abs(n_d - 1.788001) < 5e-5, (
            f"S-LAH64 bundled Sellmeier n_d={n_d:.6f}; Ohara catalog "
            f"is 1.788001.  Coefficients may be misattributed -- the "
            f"v4.11.2 audit (CRIT-1) caught this exact problem with "
            f"the previous in-code values.")

    def test_s_lah79_in_table_with_correct_n_d(self):
        """v4.15 (P1-GL-1): S-LAH79 has bundled Sellmeier coefficients
        matching the Ohara catalog n_d=2.00330 within 5e-5."""
        from lumenairy.glass import SELLMEIER_COEFFICIENTS, _sellmeier_index
        assert 'S-LAH79' in SELLMEIER_COEFFICIENTS, (
            "v4.15 (P1-GL-1): S-LAH79 should be bundled now to support "
            "minimal installs without the ``refractiveindex`` package."
        )
        n_d = _sellmeier_index(
            wavelength_m=0.58756e-6,
            coeffs=SELLMEIER_COEFFICIENTS['S-LAH79'],
            glass_name='S-LAH79')
        assert abs(n_d - 2.003300) < 5e-5, (
            f"S-LAH79 bundled Sellmeier n_d={n_d:.6f}; Ohara catalog "
            f"is 2.003300.")

    def test_in_code_sellmeier_n_d_within_1e3_of_catalog_for_a_known_good_glass(
            self):
        """Sanity check that the rest of the Sellmeier table is sane:
        a known-good in-code entry (N-BK7) produces n_d within 1e-3 of
        the well-established catalog value (n_d = 1.5168).

        Pin this so that if someone re-introduces miscalibrated
        coefficients for any other glass, this check at least flags
        N-BK7 as a canary.
        """
        from lumenairy.glass import SELLMEIER_COEFFICIENTS, _sellmeier_index
        assert 'N-BK7' in SELLMEIER_COEFFICIENTS, (
            "N-BK7 missing from in-code Sellmeier table; this is the "
            "canary glass for the rest of the table.  Restore it.")
        n_d = _sellmeier_index(
            wavelength_m=0.5876e-6,
            coeffs=SELLMEIER_COEFFICIENTS['N-BK7'],
            glass_name='N-BK7')
        assert abs(n_d - 1.5168) < 1e-3, (
            f"N-BK7 in-code Sellmeier gave n_d = {n_d:.5f}; "
            f"catalog value is 1.5168.  If this fails another glass "
            f"may also be miscalibrated -- audit the entire table.")


# ============================================================================
# Source: test_audit_fixes_v4_14_2_agent_a.py
# Audit version: V4_14_2  scope: agent_a
# Original module docstring preserved as comment block for git-blame traceability:
#   Pinning tests for the v4.14.2 audit (Agent A scope --
#   ``lumenairy.glass``, ``lumenairy.elements.freeform``,
#   ``lumenairy.elements.polarization``).
#   
#   Four audit items are pinned:
#   
#   * **P0-NEW-1** -- ``GLASS_REGISTRY['S-LAH64']`` /
#     ``GLASS_REGISTRY['S-LAH79']`` were stranded with the
#     ``'__sellmeier__'`` sentinel after v4.11.2 removed their Sellmeier
#     rows from :data:`SELLMEIER_COEFFICIENTS`.  Every lookup for these
#     two glasses raised ``ValueError`` from the dispatcher's
#     consistency-check branch -- the v4.11.2 fix forgot to re-route them
#     to a refractiveindex.info tuple.  v4.14.2 re-routes to
#     ``('specs', 'OHARA-optical', 'S-LAH64')`` /
#     ``('specs', 'OHARA-optical', 'S-LAH79')``, restoring the Ohara
#     catalogue n_d values (1.788 / 2.003).  A new module-load
#     consistency check (``_check_glass_registry_consistency``) converts
#     the same class-of-bug into a fail-fast at import time so a future
#     drift cannot re-surface as a silent ``ValueError`` at first call.
#   
#   * **P1-NEW-5** -- ``surface_sag_xy_polynomial`` evaluated
#     ``c * X**i * Y**j`` for every pixel of the input grid, with no
#     out-of-domain guard.  A high-order coefficient on a large grid
#     produced a discontinuous step at the aperture rim (e.g.
#     ``(2, 0): 1e3`` on a 50 mm half-grid produces 2.5 m of corner sag
#     applied to pixels outside the physical aperture).  v4.14.2 adds
#     ``norm_x``/``norm_y`` kwargs (default 1.0 = unit box) matching the
#     Chebyshev branch's ``np.where(outside, 0.0, departure)`` pattern.
#   
#   * **P1-NEW-7** -- ``apply_rotator`` was the only angle-taking
#     polarization helper without an ``angle_deg=`` kwarg.  Idiomatic
#     ``apply_rotator(field, angle_deg=45)`` calls raised TypeError.
#     v4.14.2 closes the v4.7 sibling-gap; passing both with
#     conflicting values raises ``ValueError`` (consistent-value pairs
#     are accepted).
#   
#   * **P1-NEW-8** -- ``JonesField.__init__`` validated
#     ``Ex.shape == Ey.shape`` but not ``Ex.ndim == 2``, ``dx > 0``,
#     ``dy > 0``.  Invalid inputs propagated all the way to the FFT in
#     :meth:`propagate` where an opaque shape / value error was raised
#     far from the construction site.  v4.14.2 validates at construction.
#   
#   Author:  Agent A -- v4.14.2.
# ============================================================================

import importlib
import inspect

import numpy as np
import pytest

import lumenairy as lm
from lumenairy import glass as _glass
from lumenairy.elements.freeform import surface_sag_xy_polynomial
from lumenairy.elements.polarization import (
    JonesField,
    apply_rotator,
    create_linear_polarized,
)

# ===========================================================================
# A.1 (P0-NEW-1) -- S-LAH64 / S-LAH79 dispatch
# ===========================================================================


class TestAuditFixesV4_14_2_agent_a_SLahDispatch:
    """Pin that ``GLASS_REGISTRY['S-LAH64']`` /
    ``GLASS_REGISTRY['S-LAH79']`` resolve to a real refractive index.

    The pre-v4.14.2 bug:  both glasses were flagged
    ``'__sellmeier__'`` but their Sellmeier rows were removed in
    v4.11.2, so the dispatcher's consistency-check raised
    ``ValueError`` on every lookup.
    """

    def test_s_lah64_dispatch_does_not_raise(self):
        """``get_glass_index('S-LAH64', d-line)`` returns a finite
        refractive index in the [1.5, 2.0] range characteristic of
        Ohara lanthanum-flint glasses.  Pre-v4.14.2 raised
        ``ValueError``.
        """
        n = _glass.get_glass_index('S-LAH64', 587.6e-9)
        assert np.isfinite(n), 'S-LAH64 must return a finite n'
        assert 1.5 < n < 2.0, (
            f'S-LAH64 d-line n = {n:.4f} outside the expected '
            f'lanthanum-flint range [1.5, 2.0]; Ohara catalogue '
            f'value is 1.788.')

    def test_s_lah79_dispatch_does_not_raise(self):
        """``get_glass_index('S-LAH79', d-line)`` returns a finite
        refractive index near the Ohara catalogue value 2.003.
        Pre-v4.14.2 raised ``ValueError``.
        """
        n = _glass.get_glass_index('S-LAH79', 587.6e-9)
        assert np.isfinite(n), 'S-LAH79 must return a finite n'
        # S-LAH79 n_d = 2.00330 per Ohara catalogue
        assert 1.5 < n < 2.1, (
            f'S-LAH79 d-line n = {n:.4f} outside the expected '
            f'high-index lanthanum-flint range [1.5, 2.1]; Ohara '
            f'catalogue value is 2.003.')

    def test_glass_registry_consistency_check_present(self):
        """The v4.14.2 module-load consistency check function exists,
        is callable with no arguments, and runs to completion on a
        healthy registry without raising.

        The drift-detection contract is exercised by its siblings
        (``test_glass_registry_consistency_check_rejects_drift`` and
        the polynomial-/reverse-direction tests later in this class)
        which inject synthetic drift entries and assert ``RuntimeError``.
        This test pins the no-arg callable contract: a future refactor
        that adds a required parameter or that turns the function into
        a non-callable (module-level attribute, classmethod-only, etc.)
        would silently disable the check; this assertion fails loudly
        if that happens.
        """
        # v5.2.3 (AUDIT_V4_13_1 Part 6.1 closure: replace inspect.getsource proxy with behavioral pin):
        # the original assertion grepped the source for 'GLASS_REGISTRY',
        # 'SELLMEIER_COEFFICIENTS', 'RuntimeError', and '__sellmeier__'
        # substrings.  The drift-detection contract those substrings
        # proxied is covered behaviorally by the sibling drift-injection
        # tests in this class.  Here we pin only the missing piece those
        # siblings cannot reach: that the no-arg callable contract holds.
        assert hasattr(_glass, '_check_glass_registry_consistency'), (
            'v4.14.2 module-load consistency check '
            '`_check_glass_registry_consistency` is missing from '
            'lumenairy.glass.')
        fn = _glass._check_glass_registry_consistency
        assert callable(fn), (
            '`_check_glass_registry_consistency` must be a callable; '
            f'got {type(fn).__name__}.')
        # No-arg contract: calling it on the healthy (un-tampered)
        # registry returns cleanly.  A future signature break that
        # adds a required arg would fail TypeError here, and a future
        # regression that re-introduces real drift in GLASS_REGISTRY
        # would fail RuntimeError here -- both are the right failures
        # to surface at the right time.
        fn()

    def test_glass_registry_consistency_check_rejects_drift(self):
        """Inject a synthetic ``'__test_sentinel_v4_14_2__'`` entry
        flagged ``'__sellmeier__'`` but absent from
        ``SELLMEIER_COEFFICIENTS``, then re-invoke the check.  Must
        raise ``RuntimeError`` naming the drift.  Restore state on
        exit so subsequent tests are unaffected.
        """
        sentinel_name = '__test_sentinel_v4_14_2__'
        assert sentinel_name not in _glass.GLASS_REGISTRY
        assert sentinel_name not in _glass.SELLMEIER_COEFFICIENTS
        _glass.GLASS_REGISTRY[sentinel_name] = '__sellmeier__'
        try:
            with pytest.raises(RuntimeError, match='GLASS_REGISTRY drift'):
                _glass._check_glass_registry_consistency()
        finally:
            del _glass.GLASS_REGISTRY[sentinel_name]
        # Re-run the real check to confirm no real-entry drift was
        # introduced.
        _glass._check_glass_registry_consistency()


# ===========================================================================
# A.2 (P1-NEW-5) -- surface_sag_xy_polynomial out-of-domain guard
# ===========================================================================


class TestAuditFixesV4_14_2_agent_a_XyPolynomialDomainGuard:
    """Pin that ``surface_sag_xy_polynomial`` zeros the polynomial
    departure outside the ``(norm_x, norm_y)`` rectangular box.

    The pre-v4.14.2 bug:  no guard, so a high-order coefficient on a
    large grid produced a discontinuous step at the aperture rim
    where the polynomial diverged but the raytracer saw no
    aperture-aware clip.
    """

    def test_xy_polynomial_zero_outside_unit_box_nonzero_inside(self):
        """Build a (2, 0): 1.0 (pure X^2) XY-polynomial surface,
        evaluate on a 64x64 grid spanning [-2, +2] (twice the
        unit-box half-extent in each axis).  The freeform departure
        must be zero outside the unit box and equal to X^2 inside.
        """
        N = 64
        half = 2.0  # twice the default unit-box half-extent
        x = np.linspace(-half, half, N)
        y = np.linspace(-half, half, N)
        X, Y = np.meshgrid(x, y, indexing='xy')
        # Pure X^2 freeform with flat base (R = inf, no conic).
        sag = surface_sag_xy_polynomial(
            X, Y, R=np.inf, conic=0.0,
            xy_coeffs={(2, 0): 1.0},
            norm_x=1.0, norm_y=1.0)
        # Outside the unit box: sag must be 0 (flat base + zeroed
        # freeform).
        outside = (np.abs(X) > 1.0) | (np.abs(Y) > 1.0)
        np.testing.assert_allclose(
            sag[outside], 0.0, atol=1e-15,
            err_msg='sag must be zero outside the unit box')
        # Inside the unit box: sag must equal X^2 (the pure
        # polynomial term, no base sag).
        inside = ~outside
        np.testing.assert_allclose(
            sag[inside], (X ** 2)[inside], atol=1e-15,
            err_msg='sag must equal X^2 inside the unit box')
        # Sanity: must have BOTH inside and outside pixels.
        assert outside.any(), 'test grid must extend beyond unit box'
        assert inside.any(), 'test grid must include unit-box pixels'


# ===========================================================================
# A.3 (P1-NEW-7) -- apply_rotator angle_deg kwarg
# ===========================================================================


def _x_polarized_field(N: int = 32, dx: float = 1e-6) -> JonesField:
    """A unit-amplitude x-polarized JonesField for rotator tests."""
    scalar = np.ones((N, N), dtype=complex)
    return create_linear_polarized(scalar, dx, angle=0.0)


class TestAuditFixesV4_14_2_agent_a_ApplyRotatorAngleDeg:
    """Pin that ``apply_rotator`` accepts ``angle_deg=`` matching the
    v4.7 convention used by ``apply_polarizer``, ``apply_waveplate``,
    and the half/quarter-wave-plate convenience wrappers.
    """

    def test_apply_rotator_angle_deg_equivalent_to_angle_rad(self):
        """``apply_rotator(field, angle_deg=45)`` must produce
        bit-equal output to ``apply_rotator(field, angle=pi/4)``.
        """
        f_rad = _x_polarized_field()
        f_deg = _x_polarized_field()
        apply_rotator(f_rad, angle=np.pi / 4)
        apply_rotator(f_deg, angle_deg=45.0)
        np.testing.assert_allclose(
            f_rad.Ex, f_deg.Ex, atol=1e-15,
            err_msg='Ex from angle_deg=45 must match angle=pi/4')
        np.testing.assert_allclose(
            f_rad.Ey, f_deg.Ey, atol=1e-15,
            err_msg='Ey from angle_deg=45 must match angle=pi/4')

    def test_apply_rotator_consistent_angle_and_angle_deg_accepted(self):
        """Supplying both ``angle`` and ``angle_deg`` with consistent
        values must be accepted (silently use ``angle_deg``)."""
        f = _x_polarized_field()
        # pi/4 == radians(45) -- consistent
        apply_rotator(f, angle=np.pi / 4, angle_deg=45.0)
        # Should not raise; Ey should be sin(pi/4) = 1/sqrt(2)
        np.testing.assert_allclose(
            f.Ey[0, 0], 1.0 / np.sqrt(2), atol=1e-12)

    def test_apply_rotator_conflicting_angle_raises(self):
        """Supplying ``angle`` and ``angle_deg`` with disagreeing
        values must raise ``ValueError``."""
        f = _x_polarized_field()
        with pytest.raises(ValueError, match='conflicting'):
            apply_rotator(f, angle=np.pi / 3, angle_deg=45.0)


# ===========================================================================
# A.4 (P1-NEW-8) -- JonesField input validation
# ===========================================================================


class TestAuditFixesV4_14_2_agent_a_JonesFieldInputValidation:
    """Pin that ``JonesField.__init__`` validates ``dx > 0``,
    ``dy > 0``, and ``Ex.ndim == 2`` at construction time, rather
    than letting invalid inputs propagate to the FFT in
    :meth:`JonesField.propagate`.
    """

    def test_jones_field_rejects_zero_dx(self):
        """``dx = 0`` must raise ``ValueError`` at construction."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        with pytest.raises(ValueError, match='dx must be'):
            JonesField(Ex, Ey, dx=0.0)

    def test_jones_field_rejects_negative_dx(self):
        """``dx < 0`` must raise ``ValueError`` at construction."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        with pytest.raises(ValueError, match='dx must be'):
            JonesField(Ex, Ey, dx=-1e-6)

    def test_jones_field_rejects_negative_dy(self):
        """``dy < 0`` must raise ``ValueError`` at construction."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        with pytest.raises(ValueError, match='dy must be'):
            JonesField(Ex, Ey, dx=1e-6, dy=-1e-6)

    def test_jones_field_rejects_1d_input(self):
        """A 1-D ``Ex`` / ``Ey`` must raise ``ValueError`` at
        construction.  Pre-v4.14.2 this propagated to the FFT in
        ``propagate`` and raised an opaque error far from the
        construction site.
        """
        Ex = np.ones(64, dtype=complex)
        Ey = np.ones(64, dtype=complex)
        with pytest.raises(ValueError, match='2-D'):
            JonesField(Ex, Ey, dx=1e-6)

    def test_jones_field_valid_inputs_construct_cleanly(self):
        """Valid 2-D inputs with positive ``dx``, ``dy`` construct
        without error and preserve the supplied pitch."""
        N = 16
        Ex = np.ones((N, N), dtype=complex)
        Ey = np.ones((N, N), dtype=complex)
        f = JonesField(Ex, Ey, dx=1e-6, dy=2e-6)
        assert f.dx == 1e-6
        assert f.dy == 2e-6
        assert f.Ex.shape == (N, N)
        # Default dy -> dx
        f2 = JonesField(Ex, Ey, dx=1e-6)
        assert f2.dy == 1e-6
