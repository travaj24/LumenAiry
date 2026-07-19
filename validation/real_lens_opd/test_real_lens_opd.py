"""Gating regression test for the real-lens wave-vs-geometric OPD check.

Closes audit finding S5-2: previously ``run_validation.py`` produced
plots and a report but had **no pass/fail threshold**, and this
directory contained no ``test_*.py`` -- so ``validation/run_all.py``'s
``rglob('test_*.py')`` never discovered it and CI never ran it.  A
systematic wave-vs-geometric OPD regression on the core real-lens path
would have tripped nothing automated.

This thin wrapper reuses :func:`run_validation.run_case` (the exact same
machinery that generates the report) on a small, fast, well-corrected
representative set and asserts that the residual between the wave OPD
(``apply_real_lens`` / ``apply_real_lens_traced``) and the **independent
oracle** -- lumenairy's own sequential vector ray tracer's geometric OPL
-- stays within physical bounds.

The oracle is genuinely independent of the wave models: the geometric
OPL comes from ``raytrace.trace`` (exact vector Snell's law, surface by
surface), while the wave OPD comes from FFT angular-spectrum phase-screen
propagation.  Bounding their difference is a real cross-check, not a
tautology -- a focal-length, OPL-bookkeeping, or slant-sign regression
in either path blows the residual well past these bounds (verified: a
200 nm rho^4 oracle mismatch pushes the raw slant RMS from ~6 nm to
~33 nm).

Grids are kept small (N=1024) so the whole file runs in a few seconds,
suitable for every-push CI.  ``run_validation.py`` remains the heavy
full-suite report generator (tens of minutes).

Run::

    python validation/real_lens_opd/test_real_lens_opd.py   # exits 0/1
    pytest validation/real_lens_opd/test_real_lens_opd.py    # test_* items
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make ``run_validation`` / ``lens_cases`` and the library importable
# regardless of how this file is loaded (as a script under run_all.py,
# or as a package module under pytest).
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..'))
for _p in (_HERE, _LIB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_validation import run_case  # noqa: E402  (script-style import)

from lumenairy.io.prescriptions import make_singlet  # noqa: E402

# -------------------------------------------------------------------------
# Physical residual bounds [nm].
#
# For a well-corrected moderate-f/# plano-convex singlet the wave OPD and
# the geometric OPL agree to a few nm RMS.  These caps are set at the
# "few tens of nm" physical scale the audit calls out, giving ~5-10x
# headroom over the correct-code values (BK7: raw 5.9, ptf 1.7, traced
# 0.5 nm) while still failing on any systematic regression.
# -------------------------------------------------------------------------
RAW_SLANT_MAX_NM = 30.0      # absolute wave-vs-geom agreement (catches
                             # focal-length / OPL-bookkeeping / sign bugs)
PTF_SLANT_MAX_NM = 15.0      # high-order residual after piston+tilt+defocus
PTF_TRACED_MAX_NM = 8.0      # same, independent per-pixel ray-traced screen
MIN_FINITE_SAMPLES = 50      # guard against a vacuously-empty comparison


def _gate_case(name, R1, R2, d, glass, aperture, wavelength, N, dx,
               description):
    """Build a small singlet case and run it through ``run_case``.

    Returns the ``run_case`` result dict.
    """
    case = {
        'name': name,
        'description': description,
        'prescription': make_singlet(R1, R2, d, glass,
                                     aperture=aperture, name=name),
        'aperture': aperture,
        'wavelength': wavelength,
        'N': N,
        'dx': dx,
        'category': 'gate',
    }
    return run_case(case, verbose=False)


def _assert_bounded(name, result):
    """Assert the wave-vs-geom residuals for one case are within bounds.

    Raises AssertionError (fails the test) on breach; prints the measured
    numbers so they show up in CI logs on pass too.
    """
    raw = result['summaries']['raw']
    ptf = result['summaries']['piston+tilt+defocus']
    raw_slant = raw['slant']['rms_nm']
    ptf_slant = ptf['slant']['rms_nm']
    ptf_traced = ptf['ray-traced']['rms_nm']
    n_finite = int(np.isfinite(result['residuals_nm']['slant']).sum())

    print(f"  [{name}] EFL={result['f_nom']*1e3:.1f} mm  "
          f"n_finite={n_finite}  "
          f"raw_slant={raw_slant:.2f} nm  "
          f"ptf_slant={ptf_slant:.2f} nm  "
          f"ptf_traced={ptf_traced:.2f} nm")

    assert n_finite > MIN_FINITE_SAMPLES, (
        f"{name}: only {n_finite} finite in-aperture samples -- the "
        f"wave/geom comparison is empty, bounds would pass vacuously")
    assert raw_slant < RAW_SLANT_MAX_NM, (
        f"{name}: raw wave-vs-geom slant RMS {raw_slant:.2f} nm exceeds "
        f"{RAW_SLANT_MAX_NM} nm -- systematic OPD regression "
        f"(focal length / OPL bookkeeping / slant sign)")
    assert ptf_slant < PTF_SLANT_MAX_NM, (
        f"{name}: piston+tilt+defocus slant RMS {ptf_slant:.2f} nm exceeds "
        f"{PTF_SLANT_MAX_NM} nm -- high-order aberration mismatch")
    assert ptf_traced < PTF_TRACED_MAX_NM, (
        f"{name}: piston+tilt+defocus ray-traced RMS {ptf_traced:.2f} nm "
        f"exceeds {PTF_TRACED_MAX_NM} nm -- per-pixel phase-screen "
        f"regression")


# -------------------------------------------------------------------------
# Test cases -- two fast, well-corrected geometries.
# -------------------------------------------------------------------------

def test_real_lens_opd_plano_convex_bk7():
    """Wave OPD matches ray-traced OPL for an f/16 N-BK7 plano-convex."""
    r = _gate_case(
        'gate_plano_convex_R50_BK7', 50e-3, np.inf, 4.0e-3, 'N-BK7',
        aperture=6.0e-3, wavelength=1.31e-6, N=1024, dx=8e-6,
        description='Plano-convex R=50 mm, N-BK7 (OPD gate)')
    _assert_bounded('plano_convex_BK7', r)


def test_real_lens_opd_plano_convex_sf6():
    """Same check on a high-index (N-SF6HT) plano-convex -- stronger
    refraction guards against index/dispersion-specific OPD bugs."""
    r = _gate_case(
        'gate_plano_convex_R50_SF6', 50e-3, np.inf, 4.0e-3, 'N-SF6HT',
        aperture=6.0e-3, wavelength=1.31e-6, N=1024, dx=8e-6,
        description='Plano-convex R=50 mm, N-SF6HT (OPD gate)')
    _assert_bounded('plano_convex_SF6', r)


# -------------------------------------------------------------------------
# Legacy run_all.py entry point: run every test_* function, exit 0/1.
# -------------------------------------------------------------------------

def main():
    print(f"Real-lens OPD gate (S5-2): "
          f"raw_slant<{RAW_SLANT_MAX_NM} ptf_slant<{PTF_SLANT_MAX_NM} "
          f"ptf_traced<{PTF_TRACED_MAX_NM} [nm]")
    tests = [
        test_real_lens_opd_plano_convex_bk7,
        test_real_lens_opd_plano_convex_sf6,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001  (report + fail, don't crash suite)
            failed += 1
            print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{'ALL PASSED' if failed == 0 else f'{failed} FAILED'}")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
