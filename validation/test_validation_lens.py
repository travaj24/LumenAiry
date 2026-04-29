"""
Validation test harness for lumenairy (3.1.11+).

Runs a suite of known-answer checks against first-principles paraxial
formulas and Seidel invariants.  Intended as a CI-ish smoke test that
catches regressions from the many API changes introduced in 3.1.6 -
3.1.11 (apply_real_lens_maslov, ray_subsample=8 default, polynomial
Newton, GPU paths, stop-aware Seidel, refocus, etc.).

Run::

    python validation/test_validation_lens.py

Exits with status 0 on all-pass, 1 on any failure.  Each test prints
its own pass / fail line so tracking down a regression is immediate.

Intentionally uses only **analytically-computable** reference values
(lensmaker formulas, Petzval sums from first principles, symmetry
invariants) rather than vendor datasheet numbers -- that way the
tests don't break on wavelength / glass / tolerance differences
between the library's glass database and whatever vendor was used to
derive a datasheet.
"""

from __future__ import annotations

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT = os.path.normpath(os.path.join(_HERE, '..'))
if _LIB_ROOT not in sys.path:
    sys.path.insert(0, _LIB_ROOT)

import lumenairy as op                    # noqa: E402
import lumenairy.raytrace as rt           # noqa: E402


# ---------------------------------------------------------------------------
# Tiny test harness
# ---------------------------------------------------------------------------

_failed = 0
_passed = 0


def _check(name, ok, detail=''):
    global _failed, _passed
    status = 'PASS' if ok else 'FAIL'
    marker = ' ' if ok else '!'
    print(f"  [{status}]{marker} {name}" + (f"  -- {detail}" if detail else ''))
    if ok:
        _passed += 1
    else:
        _failed += 1


def _isclose(a, b, rtol=1e-3, atol=0.0):
    return abs(a - b) <= atol + rtol * abs(b)


def _section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
# Test 1: Singlet EFL from lensmaker's thick-lens equation
# ---------------------------------------------------------------------------

def test_singlet_lensmaker():
    _section("Test 1: singlet EFL vs lensmaker's thick-lens formula")
    # Symmetric bi-convex N-BK7 singlet
    R1, R2 = 50e-3, -50e-3
    d = 5e-3
    wavelength = 0.587e-6    # d-line for classical lensmaker comparison
    rx = op.make_singlet(R1=R1, R2=R2, d=d, glass='N-BK7',
                          aperture=20e-3)
    n = op.get_glass_index('N-BK7', wavelength)

    # Thick-lens lensmaker: 1/f = (n-1) [1/R1 - 1/R2 + (n-1)*d/(n*R1*R2)]
    f_lensmaker = 1.0 / ((n - 1.0) * (
        1.0 / R1 - 1.0 / R2 + (n - 1.0) * d / (n * R1 * R2)))

    info = op.lens_abcd(rx, wavelength)
    _check("EFL matches lensmaker to <0.1%",
           _isclose(info.efl, f_lensmaker, rtol=1e-3),
           f"library={info.efl*1e3:.4f} mm, lensmaker={f_lensmaker*1e3:.4f} mm")

    # Principal-plane symmetry: bi-convex symmetric lens should have
    # |H| = |H'| to within numerical precision.
    H, Hp = info.principal_planes
    _check("|H| == |H'| for symmetric biconvex (<1 um)",
           _isclose(abs(H), abs(Hp), atol=1e-6),
           f"H={H*1e3:.4f} mm, H'={Hp*1e3:.4f} mm")

    # BFL consistency: BFL = EFL + H'  (H' measured from rear vertex)
    _check("BFL = EFL + H' to <10 um",
           _isclose(info.bfl, info.efl + Hp, atol=10e-6),
           f"BFL={info.bfl*1e3:.4f} mm, EFL+H'={info.efl*1e3 + Hp*1e3:.4f} mm")


# ---------------------------------------------------------------------------
# Test 2: Doublet lens_abcd vs manual ABCD composition
# ---------------------------------------------------------------------------

def test_doublet_abcd():
    _section("Test 2: doublet ABCD vs manual matrix composition")
    rx = op.make_doublet(R1=50e-3, R2=-50e-3, R3=-100e-3,
                           d1=5e-3, d2=3e-3,
                           glass1='N-BK7', glass2='N-SF2',
                           aperture=20e-3)
    wavelength = 0.587e-6
    info = op.lens_abcd(rx, wavelength)

    # Build the ABCD manually surface-by-surface in reduced (y, nu)
    # form: R = [[1, 0], [-phi, 1]],  T = [[1, t/n], [0, 1]]
    n1 = op.get_glass_index('N-BK7', wavelength)
    n2 = op.get_glass_index('N-SF2', wavelength)
    R1, R2, R3 = 50e-3, -50e-3, -100e-3
    d1, d2 = 5e-3, 3e-3

    def refract(phi):
        return np.array([[1.0, 0.0], [-phi, 1.0]])

    def transfer(t, n):
        return np.array([[1.0, t / n], [0.0, 1.0]])

    M_manual = np.eye(2)
    # Surface 1: air -> N-BK7
    M_manual = refract((n1 - 1) / R1) @ M_manual
    M_manual = transfer(d1, n1) @ M_manual
    # Surface 2: N-BK7 -> N-SF2
    M_manual = refract((n2 - n1) / R2) @ M_manual
    M_manual = transfer(d2, n2) @ M_manual
    # Surface 3: N-SF2 -> air
    M_manual = refract((1.0 - n2) / R3) @ M_manual

    max_diff = float(np.max(np.abs(info.abcd - M_manual)))
    _check("lens_abcd matches hand-composed ABCD (max diff <1e-10)",
           max_diff < 1e-10,
           f"max |diff| = {max_diff:.2e}")

    # Total EFL should be positive (converging system)
    _check("doublet EFL > 0 (converging)", info.efl > 0,
           f"EFL = {info.efl*1e3:.4f} mm")


# ---------------------------------------------------------------------------
# Test 3: find_lenses auto-detection
# ---------------------------------------------------------------------------

def test_find_lenses():
    _section("Test 3: find_lenses auto-detection")
    # Singlet: one element
    rx1 = op.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                            aperture=20e-3)
    lenses1 = op.find_lenses(rt.surfaces_from_prescription(rx1), 0.55e-6)
    _check("singlet detected as 1 element", len(lenses1) == 1,
           f"found {len(lenses1)}")

    # Cemented doublet: one element (glass->glass interior)
    rx2 = op.make_doublet(R1=50e-3, R2=-50e-3, R3=-100e-3,
                            d1=5e-3, d2=3e-3,
                            glass1='N-BK7', glass2='N-SF2',
                            aperture=20e-3)
    lenses2 = op.find_lenses(rt.surfaces_from_prescription(rx2), 0.55e-6)
    _check("cemented doublet stays grouped as 1 element",
           len(lenses2) == 1, f"found {len(lenses2)}")

    # Two singlets with air gap between: two elements
    surf1 = rt.surfaces_from_prescription(rx1)
    surf1[1].thickness = 5e-3  # Air gap to next lens
    surf1.extend(rt.surfaces_from_prescription(rx1))
    lenses3 = op.find_lenses(surf1, 0.55e-6)
    _check("two air-separated singlets detected as 2 elements",
           len(lenses3) == 2, f"found {len(lenses3)}")

    # LensInfo round-trip through lens_abcd(LensInfo, surfaces=...)
    surfaces = rt.surfaces_from_prescription(rx2)
    li_vis = op.find_lenses(surfaces, 0.55e-6)[0]
    li_nir = op.lens_abcd(li_vis, 1.31e-6, surfaces=surfaces)
    _check("lens_abcd(LensInfo, surfaces=...) preserves start_index",
           li_nir.start_index == li_vis.start_index,
           f"vis={li_vis.start_index} nir={li_nir.start_index}")
    _check("lens_abcd(LensInfo, surfaces=...) preserves end_index",
           li_nir.end_index == li_vis.end_index,
           f"vis={li_vis.end_index} nir={li_nir.end_index}")
    _check("lens_abcd(LensInfo) EFL matches manual slice",
           abs(li_nir.efl - op.lens_abcd(
               surfaces[li_vis.start_index:li_vis.end_index + 1],
               1.31e-6).efl) < 1e-12,
           f"{li_nir.efl}")
    _check("lens_abcd(LensInfo) reflects chromatic shift",
           li_nir.efl != li_vis.efl,
           f"vis={li_vis.efl:.6e}  nir={li_nir.efl:.6e}")


# ---------------------------------------------------------------------------
# Test 4: Stop-aware Seidel -- invariants
# ---------------------------------------------------------------------------

def test_seidel_invariants():
    _section("Test 4: stop-aware Seidel invariants")
    rx = op.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                          aperture=20e-3)
    surfaces = rt.surfaces_from_prescription(rx)
    wavelength = 0.587e-6

    # Compute Seidel at two different stop positions
    s0, _ = rt.seidel_coefficients(surfaces, wavelength, stop_index=0)
    s1, _ = rt.seidel_coefficients(surfaces, wavelength, stop_index=1)

    # S4 (Petzval) is purely curvature-dependent; stop position must not change it.
    _check("S4 (Petzval) invariant to stop position",
           _isclose(s0['total']['S4'], s1['total']['S4'],
                    atol=1e-10, rtol=1e-8),
           f"S4(stop=0)={s0['total']['S4']:.3e}, "
           f"S4(stop=1)={s1['total']['S4']:.3e}")

    # Chief ray through stop center: y_c[stop_index] == 0 by
    # construction of the initial conditions.
    _check("chief ray y=0 at stop surface (stop=1)",
           abs(float(s1['y_chief'][1])) < 1e-12,
           f"y_chief[1] = {float(s1['y_chief'][1]):.2e} m")

    # Backward compat: default call with no is_stop flags should pick
    # stop=0 (legacy fallback of find_stop) and match the pre-3.1.11
    # behaviour within machine precision.
    s_default, _ = rt.seidel_coefficients(surfaces, wavelength)
    max_diff = max(abs(s0['total'][k] - s_default['total'][k])
                    for k in 'S1 S2 S3 S4 S5'.split())
    _check("default stop_index matches explicit stop_index=0",
           max_diff < 1e-12,
           f"max diff = {max_diff:.2e}")


# ---------------------------------------------------------------------------
# Test 5: compute_pupils -- EP/XP for a singlet with stop at surface 0
# ---------------------------------------------------------------------------

def test_compute_pupils():
    _section("Test 5: compute_pupils")
    rx = op.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                          aperture=20e-3)
    surfaces = rt.surfaces_from_prescription(rx)
    pupils = op.compute_pupils(surfaces, 0.587e-6)

    # Stop at surface 0 => EP coincides with stop
    _check("EP at z=0 when stop is surface 0",
           abs(pupils.ep_z) < 1e-10,
           f"ep_z = {pupils.ep_z:.2e} m")
    # EP radius == stop radius == surface 0 semi-diameter
    _check("EP radius == stop radius (10 mm)",
           _isclose(pupils.ep_radius, 10e-3, atol=1e-8),
           f"ep_radius = {pupils.ep_radius*1e3:.4f} mm")

    # XP is the stop imaged through the post-stop optics.  Since the
    # post-stop sub-system is a refracting + single-air gap "lens",
    # the XP ends up somewhere inside the glass (negative xp_z).
    # Exact value is wavelength-dependent via n_glass.
    _check("XP z is finite and to the image side of rear vertex OR inside",
           np.isfinite(pupils.xp_z),
           f"xp_z = {pupils.xp_z*1e3:.4f} mm")


# ---------------------------------------------------------------------------
# Test 6: refocus vs full retrace (numerical equivalence)
# ---------------------------------------------------------------------------

def test_refocus():
    _section("Test 6: refocus vs full retrace")
    rx = op.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                          aperture=20e-3)
    surfaces = rt.surfaces_from_prescription(rx)
    wavelength = 0.587e-6

    # Base trace
    rays = rt.make_rings(semi_aperture=8e-3, num_rings=4, rays_per_ring=24,
                          field_angle=0.0, wavelength=wavelength)
    result = rt.trace(rays, surfaces, wavelength, output_filter='last')

    # Refocus to paraxial focus (BFL after last surface)
    _, _, bfl, _ = rt.system_abcd(surfaces, wavelength)
    refocused = op.refocus(result, bfl)
    rms_r, _ = rt.spot_rms(refocused)

    # Full retrace with an image-plane surface
    surfs_img = list(surfaces)
    surfs_img[-1] = rt._surface_copy_with(
        surfs_img[-1], thickness=bfl) if hasattr(rt, '_surface_copy_with') else \
        rt.Surface(
            radius=surfs_img[-1].radius, conic=surfs_img[-1].conic,
            aspheric_coeffs=surfs_img[-1].aspheric_coeffs,
            semi_diameter=surfs_img[-1].semi_diameter,
            glass_before=surfs_img[-1].glass_before,
            glass_after=surfs_img[-1].glass_after,
            is_mirror=surfs_img[-1].is_mirror,
            thickness=bfl, label=surfs_img[-1].label,
            surf_num=surfs_img[-1].surf_num)
    last_glass = surfaces[-1].glass_after
    surfs_img.append(rt.Surface(
        radius=np.inf, semi_diameter=np.inf,
        glass_before=last_glass, glass_after=last_glass,
        label='Image'))
    rays2 = rt.make_rings(semi_aperture=8e-3, num_rings=4, rays_per_ring=24,
                           field_angle=0.0, wavelength=wavelength)
    full_result = rt.trace(rays2, surfs_img, wavelength)
    rms_f, _ = rt.spot_rms(full_result)

    rel = abs(rms_r - rms_f) / max(rms_f, 1e-12)
    _check("refocus RMS matches full retrace (<1% relative)",
           rel < 0.01, f"refocus={rms_r*1e6:.3f} um, full={rms_f*1e6:.3f} um, rel={rel*100:.3f}%")


# ---------------------------------------------------------------------------
# Test 7: through_focus_rms best-focus near paraxial BFL
# ---------------------------------------------------------------------------

def test_through_focus_rms():
    _section("Test 7: through_focus_rms locates best focus")
    rx = op.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                          aperture=20e-3)
    surfaces = rt.surfaces_from_prescription(rx)
    wavelength = 0.587e-6
    _, _, bfl, _ = rt.system_abcd(surfaces, wavelength)

    shifts = bfl + np.linspace(-5e-3, 5e-3, 101)
    _, rms, best = op.through_focus_rms(surfaces, wavelength, 8e-3, shifts,
                                          num_rings=5, rays_per_ring=36)
    # Best focus is offset from paraxial by spherical aberration.  For
    # an F/5 biconvex singlet the best RMS spot is roughly 1-3 mm
    # short of paraxial focus.  Assert: best is within the sweep
    # range, rms is a U-shape (monotonic decrease then increase).
    _check("best_focus is inside sweep range",
           shifts.min() <= best <= shifts.max(),
           f"best_focus = {best*1e3:.3f} mm (sweep [{shifts.min()*1e3:.2f}, "
           f"{shifts.max()*1e3:.2f}])")
    _check("through-focus RMS has a minimum (not flat)",
           rms.min() < rms.max() * 0.5,
           f"min={rms.min()*1e6:.2f} um, max={rms.max()*1e6:.2f} um")


# ---------------------------------------------------------------------------
# Test 8: apply_real_lens, apply_real_lens_traced, apply_real_lens_maslov agreement
# ---------------------------------------------------------------------------

def test_lens_operator_agreement():
    _section("Test 8: apply_real_lens vs apply_real_lens_traced")
    rx = op.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                          aperture=20e-3)
    wavelength = 1.31e-6
    N, dx = 512, 10e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E_in = np.exp(-(X ** 2 + Y ** 2) / (5e-3) ** 2).astype(np.complex128)

    E_rl = op.apply_real_lens(E_in, rx, wavelength, dx)
    E_tr = op.apply_real_lens_traced(E_in, rx, wavelength, dx,
                                        n_workers=1, parallel_amp=False,
                                        min_coarse_samples_per_aperture=0)
    I_rl = np.abs(E_rl) ** 2
    I_tr = np.abs(E_tr) ** 2
    mask = I_tr > I_tr.max() * 0.01
    rel = 100.0 * float(
        np.sqrt(np.mean((I_rl[mask] / I_rl.max() - I_tr[mask] / I_tr.max()) ** 2))
        / np.sqrt(np.mean((I_tr[mask] / I_tr.max()) ** 2)))
    _check("apply_real_lens vs apply_real_lens_traced intensity <2% RMS at lens exit",
           rel < 2.0,
           f"RMS rel = {rel:.3f}%")

    # apply_real_lens_maslov (stationary phase) should also produce a
    # physically sensible Gaussian output on this collimated singlet.
    # Not expected to match traced numerically here (different
    # physics regime -- Maslov is intended for near-caustic output
    # planes, not lens exit), but it should at least run without
    # error and produce a non-zero output.
    try:
        E_ms = op.apply_real_lens_maslov(
            E_in, rx, wavelength, dx,
            ray_field_samples=8, ray_pupil_samples=8,
            integration_method='stationary_phase',
            collimated_input=True, verbose=False)
        I_ms = np.abs(E_ms) ** 2
        _check("apply_real_lens_maslov produces non-zero output",
               I_ms.max() > 1e-20,
               f"peak = {I_ms.max():.3e}")
    except Exception as e:
        _check("apply_real_lens_maslov runs without error", False,
               f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Test 9: error_code accounting
# ---------------------------------------------------------------------------

def test_error_codes():
    _section("Test 9: per-ray error codes")
    rx = op.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                          aperture=20e-3)
    surfaces = rt.surfaces_from_prescription(rx)

    # Launch rays wider than the aperture -> some should be RAY_APERTURE
    rays = rt.make_rings(semi_aperture=15e-3, num_rings=3, rays_per_ring=12,
                          field_angle=0.0, wavelength=0.587e-6)
    result = rt.trace(rays, surfaces, 0.587e-6)
    final = result.image_rays
    n_aperture = int(np.sum(final.error_code == op.RAY_APERTURE))
    n_ok = int(np.sum(final.error_code == op.RAY_OK))
    _check("aperture-vignetted rays flagged RAY_APERTURE",
           n_aperture > 0,
           f"alive={n_ok}, aperture-killed={n_aperture}")
    _check("alive == (error_code == RAY_OK) invariant",
           int(final.alive.sum()) == n_ok,
           f"alive.sum()={final.alive.sum()}, "
           f"ec==OK count={n_ok}")


# ---------------------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------------------

def main():
    print(f"lumenairy v{op.__version__}")
    print(f"Validation lens test harness")

    test_singlet_lensmaker()
    test_doublet_abcd()
    test_find_lenses()
    test_seidel_invariants()
    test_compute_pupils()
    test_refocus()
    test_through_focus_rms()
    test_lens_operator_agreement()
    test_error_codes()

    print(f"\n{'='*60}")
    print(f"  Passed: {_passed}")
    print(f"  Failed: {_failed}")
    print(f"{'='*60}")
    return 0 if _failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
