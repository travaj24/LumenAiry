"""v4.16.1 / Agent A -- four real-correctness bug closures.

Pins the four AUDIT_V4_16_0_DEEP_2026_05_19 P1 findings that the
prior verification-style audits missed.  Each is a silent-wrong-
answer at user-relevant configurations.

* **Bug 1** (``optimize/core.py``) -- ``MultiWavelengthMerit.evaluate``
  SUMS rather than averages while its docstring documents "average"
  semantics and both sibling classes (:class:`MultiFieldMerit`,
  :class:`ToleranceAwareMerit`) divide by ``len(...)`` at their
  return.  Pre-v4.16.1 a 3-wavelength configuration silently
  tripled the chromatic merit weight relative to 1-wavelength.

* **Bug 2** (``analysis/detector.py``) -- ``shack_hartmann`` quantises
  the lenslet pitch to integer pixels (``sa_pixels = int(round(
  lenslet_pitch / dx))``) but the slope-to-wavefront integration step
  uses the REQUESTED ``lenslet_pitch`` rather than the actual on-grid
  ``sa_pixels * dx``.  Physics: the slopes are measured between
  sub-aperture centres spaced by ``sa_pixels * dx``; the integration
  formula ``delta_phi = slope * pitch`` MUST use the same pitch as
  the measurement geometry, else the reconstructed wavefront
  amplitude is biased by ``(sa_pixels * dx) / lenslet_pitch``.  For
  ``lenslet_pitch / dx = 1.7`` the bias is ~18%.

* **Bug 3** (``io/storage.py``) -- ``_detect_backend`` misclassifies
  any pre-existing directory as Zarr (``path.endswith('.zarr') or
  os.path.isdir(path)``).  A stale ``output.h5/`` directory next to
  ``output.h5`` file silently re-routes the backend.  Also raises
  ``AttributeError`` on ``pathlib.Path`` inputs despite docstring
  advertising Path support.

* **Bug 4** (``optimize/core.py`` LM branch) -- ``[b[0] if b else
  -np.inf for b in bounds]`` is bug-by-Python-truthiness:
  ``b = (None, 1.0)`` is non-empty and truthy, so ``b[0] = None``
  leaks into ``np.array(...)`` producing an object-dtype array that
  scipy's ``least_squares`` rejects with an opaque downstream
  error.  scipy's ``bounds=(lb, ub)`` spec requires each endpoint
  to be either a finite float or +/-inf, not ``None``.

Test taxonomy:
  * Bug 1: 1 regression + 1 sibling-parity test (3 sibling classes).
  * Bug 2: 1 regression (non-integer ratio) + 1 sanity (integer
    ratio still works) + 1 amplitude-ratio physics pin.
  * Bug 3: 1 regression (ambiguous .h5 + .h5/) + 1 regression
    (zarr.json marker) + 1 regression (Path input).
  * Bug 4: 1 regression (None bounds) + 1 sanity (finite bounds
    still respected).
  * 1 meta-test: fix-lines present (defensive against accidental
    revert).

Total: 11 tests.

Author: Andrew Traverso -- v4.16.1 / Agent A
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pytest

# ============================================================================
# Bug 1 -- ``MultiWavelengthMerit.evaluate`` SUM -> AVG
# ============================================================================


class _ConstantMerit:
    """Trivial sub-merit that returns a constant for every call.

    Lets us isolate the wrapper's AVG-vs-SUM behaviour without
    coupling to the per-wavelength wave-leg propagation.  Pure-
    geometric (no wave) so the wrapper's ``apply_real_lens`` branch
    is skipped entirely.
    """
    name = 'Constant'
    needs_wave = False
    weight = 1.0

    def __init__(self, value: float):
        self.value = float(value)

    def evaluate(self, ctx) -> float:
        return self.value


def _build_singlet_ctx(wl: float):
    """Minimal :class:`EvaluationContext` for the SUM/AVG regression.

    A trivial singlet prescription gives the wrapper a working
    ``surfaces_from_prescription`` / ``system_abcd`` path so the
    ABCD sentinel branch is NOT taken; the sub-merit then evaluates
    once per wavelength under the standard path.
    """
    import lumenairy as la
    from lumenairy.optimize.core import EvaluationContext

    pres = la.make_singlet(
        R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
        aperture=12e-3)
    return EvaluationContext(
        prescription=pres, wavelength=wl,
        N=16, dx=10e-6, efl=0.1, bfl=0.1)


def test_bug1_multiwavelength_merit_averages_not_sums():
    """Pin the post-v4.16.1 averaging semantics.

    Physics: the docstring says "evaluate a sub-merit at multiple
    wavelengths AND AVERAGE."  Both sibling wrapper classes return
    ``self.weight * total / max(len(...), 1)``.  Pre-v4.16.1 the
    return was ``self.weight * total`` (no normalisation), so
    adding wavelengths silently inflated the merit value
    proportional to ``n_wavelengths``.

    Build two ``MultiWavelengthMerit`` instances with the SAME
    per-wavelength constant value: one with 1 wavelength, one with
    3.  Under AVG semantics both return the same value (the
    constant times weight).  Under the old SUM semantics the 3-
    wavelength variant would return 3x the 1-wavelength variant.

    This test FAILS pre-v4.16.1 and PASSES post-v4.16.1.
    """
    from lumenairy.optimize.core import MultiWavelengthMerit

    const_value = 0.7  # arbitrary; weight=1 so result == value
    sub = _ConstantMerit(value=const_value)

    wls_1 = [1.30e-6]
    wls_3 = [1.30e-6, 1.55e-6, 1.064e-6]

    merit_1 = MultiWavelengthMerit(
        wavelengths=wls_1, sub_merit=sub, weight=1.0)
    merit_3 = MultiWavelengthMerit(
        wavelengths=wls_3, sub_merit=sub, weight=1.0)

    ctx1 = _build_singlet_ctx(wls_1[0])
    ctx3 = _build_singlet_ctx(wls_3[0])

    v1 = merit_1.evaluate(ctx1)
    v3 = merit_3.evaluate(ctx3)

    # AVG semantics: both should equal const_value (weight=1).
    assert v1 == pytest.approx(const_value, rel=1e-12), (
        f"1-wavelength merit returned {v1!r}, expected "
        f"{const_value!r} (weight=1, constant sub-merit).")
    assert v3 == pytest.approx(const_value, rel=1e-12), (
        f"3-wavelength merit returned {v3!r}, expected "
        f"{const_value!r} under AVG semantics.  Pre-v4.16.1 this "
        f"returned {3 * const_value!r} (SUM, not AVG); the bug "
        f"silently tripled the chromatic merit weight when a 3rd "
        f"wavelength was added.")

    # Hard pin against accidental SUM-revert: the 3-wavelength
    # value MUST NOT be 3x the 1-wavelength value.
    assert not np.isclose(v3, 3 * v1, rtol=1e-6), (
        f"3-wavelength merit is exactly 3x the 1-wavelength merit "
        f"({v3!r} == 3 * {v1!r}); this is the pre-v4.16.1 SUM "
        f"behaviour.  The fix should divide by len(wavelengths).")


def test_bug1_multiwavelength_merit_sibling_parity():
    """Pin the AVG-shape parity across the 3 wrapper classes.

    All three wrappers (:class:`MultiWavelengthMerit`,
    :class:`MultiFieldMerit`, :class:`ToleranceAwareMerit`) average
    their sub-merit values across the per-element loop.  This test
    pins the post-v4.16.1 shape:

      MWM.evaluate(ctx, 3 wls)  ==  const_value * weight
      MFM.evaluate(ctx, 3 angles) == const_value * weight
      TAM.evaluate(ctx, 3 trials) == const_value * weight

    All three returns must equal the (sub_merit_value * weight)
    constant; if any one of them sums instead the assertion fires.
    """
    import lumenairy as la
    from lumenairy.optimize.core import (
        EvaluationContext,
        MultiFieldMerit,
        MultiWavelengthMerit,
        ToleranceAwareMerit,
    )

    const_value = 0.3
    sub = _ConstantMerit(value=const_value)

    pres = la.make_singlet(
        R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
        aperture=12e-3)
    ctx = EvaluationContext(
        prescription=pres, wavelength=1.30e-6,
        N=16, dx=10e-6, efl=0.1, bfl=0.1)

    # MWM with 3 wavelengths.
    mwm = MultiWavelengthMerit(
        wavelengths=[1.30e-6, 1.55e-6, 1.064e-6],
        sub_merit=sub, weight=1.0)
    v_mwm = mwm.evaluate(ctx)

    # MFM with 3 field angles.  Use a separate sub instance to
    # avoid any potential statefulness in _ConstantMerit (none
    # exists today but defensive).
    sub_field = _ConstantMerit(value=const_value)
    sub_field.needs_wave = False  # avoid wave-leg costs
    mfm = MultiFieldMerit(
        field_angles=[(0.0, 0.0), (1e-3, 0.0), (0.0, 1e-3)],
        sub_merit=sub_field, weight=1.0)
    # MFM's evaluate path constructs the tilted plane wave inside
    # the loop; we don't care about correctness of that path,
    # just that the wrapper averages.  The _ConstantMerit returns
    # the constant regardless of the wave-field input.
    v_mfm = mfm.evaluate(ctx)

    # TAM with 3 trials, no perturbations (degenerate but well-
    # defined: each trial evaluates the same sub_merit constant).
    sub_tol = _ConstantMerit(value=const_value)
    tam = ToleranceAwareMerit(
        sub_merit=sub_tol,
        perturbation_spec=[{'surface_index': 0, 'decenter_std': 0.0,
                            'tilt_std': 0.0, 'form_error_rms': 0.0}],
        n_trials=3, seed=42, weight=1.0)
    v_tam = tam.evaluate(ctx)

    assert v_mwm == pytest.approx(const_value, rel=1e-9), (
        f"MultiWavelengthMerit returned {v_mwm!r}, expected "
        f"{const_value!r} under AVG semantics.")
    assert v_mfm == pytest.approx(const_value, rel=1e-9), (
        f"MultiFieldMerit returned {v_mfm!r}, expected "
        f"{const_value!r}; sibling-class AVG-shape parity broken.")
    assert v_tam == pytest.approx(const_value, rel=1e-9), (
        f"ToleranceAwareMerit returned {v_tam!r}, expected "
        f"{const_value!r}; sibling-class AVG-shape parity broken.")


# ============================================================================
# Bug 2 -- ``shack_hartmann`` wavefront pitch quantisation
# ============================================================================


def _run_tilt_sh(lenslet_pitch: float, dx: float,
                 N: int = 256,
                 wavelength: float = 633e-9,
                 lenslet_focal: float = 5e-3,
                 tilt_x: float = 1e-3):
    """Build a known linear-tilt wavefront ``phi(x) = tilt_x * x``,
    propagate through Shack-Hartmann, and return diagnostic
    quantities.

    Physics: a pure x-tilt OPD ``phi = tilt_x * X`` has slope
    ``s_x = tilt_x`` everywhere (sign / scale determined by the SH
    centroid convention; we recover it empirically from the returned
    ``slopes_x``).  The SH reconstruction is
    ``wavefront = 0.5 * (cumsum(s_x, axis=1) * pitch +
    cumsum(s_y, axis=0) * pitch)``.

    For a pure x-tilt: ``s_y = 0``, ``s_x = constant``.  Therefore
    ``wf[iy, ix] - wf[iy, ix-1] = 0.5 * s_x * pitch`` (only the
    cumsum-along-x leg contributes to the column step).  This step
    is INDEPENDENT of any complications in the centroid measurement
    -- once we know the measured ``s_x``, we can verify the
    multiplier IS the post-fix ``sa_pixels * dx`` and NOT the
    pre-fix ``lenslet_pitch``.

    Returns
    -------
    wf, slopes_x, slopes_y, sa_pixels, pitch_actual, lenslet_pitch
    """
    from lumenairy.analysis.detector import shack_hartmann

    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    opd_in = tilt_x * X
    phase = 2 * np.pi * opd_in / wavelength
    E = np.exp(1j * phase).astype(np.complex128)

    sa_pixels = int(round(lenslet_pitch / dx))
    pitch_actual = sa_pixels * dx

    sx, sy, wf, _, _ = shack_hartmann(
        E, dx, wavelength, lenslet_pitch, lenslet_focal,
        detector_pixels_per_lenslet=4)
    return wf, sx, sy, sa_pixels, pitch_actual


def _measured_pitch_from_wf(wf, slopes_x):
    """Empirically recover the multiplicative pitch used inside
    ``shack_hartmann`` by inverting the column-step relationship.

    Given the post-cumsum wavefront ``wf`` and the underlying
    ``slopes_x`` (both returned by ``shack_hartmann``), the
    reconstruction is:

        wf[iy, ix] = 0.5 * (sum_{i<=ix} slopes_x[iy, i] * P
                            + sum_{j<=iy} slopes_y[j, ix] * P)
                     -- normalized so wf[0, 0] = 0

    where ``P`` is the multiplier under test.  For a pure-x-tilt
    input the slopes_y leg is zero, so::

        wf[iy, ix+1] - wf[iy, ix] = 0.5 * slopes_x[iy, ix+1] * P

    The OOB sentinels at the edge lenslets are NaN, so we look at
    a valid interior column-step.  Returns the median across all
    valid interior (iy, ix) entries where both ``slopes_x[iy, ix+1]``
    and the adjacent ``wf`` values are finite.
    """
    nLy, nLx = wf.shape
    if nLx < 2:
        return np.nan
    # Compute the column-step at each (iy, ix-1 -> ix) for ix>=1.
    # The cumsum-along-x of slopes_x gives wf_x[iy, ix] =
    # sum_{i=0..ix} slopes_x[iy, i] * P; subtracting wf_x[iy, ix-1]
    # leaves slopes_x[iy, ix] * P.  After the 0.5 * (wf_x + wf_y)
    # average and the corner-anchor, the per-column step from a
    # pure-x-tilt is 0.5 * slopes_x[iy, ix] * P.
    wf_diff = np.diff(wf, axis=1)  # shape (nLy, nLx - 1)
    # The aligned slope value for each diff is slopes_x[:, 1:].
    sx_at = slopes_x[:, 1:]
    mask = (np.isfinite(wf_diff) & np.isfinite(sx_at)
            & (np.abs(sx_at) > 1e-20))
    if not np.any(mask):
        return np.nan
    P_vals = 2.0 * wf_diff[mask] / sx_at[mask]
    return float(np.median(P_vals))


def test_bug2_shack_hartmann_non_integer_pitch_amplitude():
    """Pin the post-fix multiplicative pitch at non-integer ratio.

    Physics check: the cumsum-style integration step has a
    multiplicative pitch ``P``.  Pre-v4.16.1 ``P = lenslet_pitch``
    (the user-requested value).  Post-v4.16.1 ``P = sa_pixels * dx``
    (the actual on-grid pitch).  For ``lenslet_pitch / dx = 1.7``
    the two values differ by ``2/1.7 = 1.176`` (~18%).

    The slopes ARE measured between sub-aperture centres spaced by
    ``sa_pixels * dx``, so the on-grid pitch is the physically
    correct integration step.  This test recovers ``P`` empirically
    from the cumsum-step relationship and asserts ``P ==
    sa_pixels * dx`` to within 1%.

    This test FAILS pre-v4.16.1 (recovered P == lenslet_pitch =
    8.5 um) and PASSES post-v4.16.1 (P == sa_pixels * dx = 10 um).
    """
    dx = 5e-6
    lenslet_pitch = 1.7 * dx  # 8.5 um, non-integer ratio -> sa_pixels=2
    wf, sx, sy, sa_pixels, pitch_actual = _run_tilt_sh(
        lenslet_pitch=lenslet_pitch, dx=dx, N=256,
        tilt_x=1e-3)
    assert sa_pixels == 2
    assert pitch_actual == pytest.approx(2.0 * dx, rel=1e-12)
    assert pitch_actual != lenslet_pitch  # the non-integer-ratio case

    P_measured = _measured_pitch_from_wf(wf, sx)
    assert np.isfinite(P_measured), (
        "could not recover P from wf/slopes (slope or wf "
        "degenerate).  Check input tilt magnitude.")

    # Post-fix: P must equal sa_pixels * dx within 1%.
    rel_post = abs(P_measured - pitch_actual) / pitch_actual
    # Pre-fix: P would have equalled lenslet_pitch.  Confirm we are
    # NOT close to the pre-fix value.
    rel_pre = abs(P_measured - lenslet_pitch) / lenslet_pitch

    assert rel_post < 0.01, (
        f"recovered P = {P_measured*1e6:.4f} um does NOT match the "
        f"post-fix expected pitch_actual = {pitch_actual*1e6:.4f} um "
        f"to within 1% (rel error {rel_post*100:.2f}%).  "
        f"sa_pixels={sa_pixels}, requested lenslet_pitch="
        f"{lenslet_pitch*1e6:.3f} um.")
    assert rel_pre > 0.15, (
        f"recovered P = {P_measured*1e6:.4f} um IS still close to "
        f"the pre-fix lenslet_pitch ({lenslet_pitch*1e6:.3f} um, rel "
        f"diff {rel_pre*100:.2f}%).  The fix may not be active.  "
        f"Post-v4.16.1 P should equal sa_pixels*dx = "
        f"{pitch_actual*1e6:.3f} um, NOT lenslet_pitch.")


def test_bug2_shack_hartmann_integer_pitch_sanity():
    """Pin: integer-ratio case (sa_pixels*dx == lenslet_pitch) is
    unchanged by the fix.

    At ``lenslet_pitch / dx = 2.0`` the pre-v4.16.1 bug had no
    visible effect because ``sa_pixels * dx`` IS ``lenslet_pitch``.
    The fix is a no-op at integer ratios; verify nothing regressed.
    """
    dx = 5e-6
    lenslet_pitch = 2.0 * dx  # 10 um, exact integer ratio
    wf, sx, sy, sa_pixels, pitch_actual = _run_tilt_sh(
        lenslet_pitch=lenslet_pitch, dx=dx, N=256,
        tilt_x=1e-3)
    assert sa_pixels == 2
    assert pitch_actual == lenslet_pitch  # exact match: integer ratio

    P_measured = _measured_pitch_from_wf(wf, sx)
    assert np.isfinite(P_measured)
    rel = abs(P_measured - pitch_actual) / pitch_actual
    assert rel < 0.01, (
        f"Integer-ratio sanity check failed: recovered P = "
        f"{P_measured*1e6:.4f} um vs expected {pitch_actual*1e6:.4f} "
        f"um (rel err {rel*100:.2f}%).  Integer-ratio is the "
        f"easiest configuration and should always work.")


def test_bug2_shack_hartmann_amplitude_ratio_physics_pin():
    """Pin the on-grid-pitch scaling across two different requested
    lenslet_pitch values that round to the same ``sa_pixels``.

    Two non-integer-ratio configurations with the SAME ``sa_pixels``
    have IDENTICAL on-grid pitch.  Post-v4.16.1 the integration
    step is ``sa_pixels * dx`` for both, so the recovered P
    matches.  Pre-v4.16.1 the integration step was the requested
    ``lenslet_pitch``, which differed by ``2.4/1.6 = 1.5x`` between
    the two configurations.

    This test is the analytical pin for the fix.
    """
    dx = 5e-6
    # Both ratios round to sa_pixels = 2 on this grid.
    wf_a, sx_a, _, sa_a, pitch_a = _run_tilt_sh(
        lenslet_pitch=1.6 * dx, dx=dx, N=256, tilt_x=1e-3)
    wf_b, sx_b, _, sa_b, pitch_b = _run_tilt_sh(
        lenslet_pitch=2.4 * dx, dx=dx, N=256, tilt_x=1e-3)
    assert sa_a == sa_b == 2
    assert pitch_a == pytest.approx(pitch_b, rel=1e-12)

    P_a = _measured_pitch_from_wf(wf_a, sx_a)
    P_b = _measured_pitch_from_wf(wf_b, sx_b)
    assert np.isfinite(P_a) and np.isfinite(P_b), (
        f"could not recover P (P_a={P_a}, P_b={P_b}).")

    # Post-fix: both must equal pitch_actual = sa_pixels*dx (10 um).
    rel = abs(P_a - P_b) / max(abs(P_a), abs(P_b), 1e-30)
    assert rel < 0.01, (
        f"Two SH configurations sharing the same sa_pixels (={sa_a}) "
        f"produced DIFFERENT recovered pitches "
        f"(P_a={P_a*1e6:.4f} um, P_b={P_b*1e6:.4f} um, |delta|/max="
        f"{rel*100:.2f}%).  Post-v4.16.1 both use the on-grid pitch "
        f"({pitch_a*1e6:.3f} um); pre-v4.16.1 they would track the "
        f"two different requested lenslet_pitch values (1.6*dx="
        f"{1.6*dx*1e6:.3f} vs 2.4*dx={2.4*dx*1e6:.3f}, ratio 1.5x).")


# ============================================================================
# Bug 3 -- ``_detect_backend`` directory misclassification
# ============================================================================


def test_bug3_detect_backend_h5_with_sibling_directory():
    """A file ``output.h5`` next to a (stale) ``output.h5/`` directory
    must dispatch to HDF5, not Zarr.

    Pre-v4.16.1: ``os.path.isdir(path)`` returned True for the
    directory variant, silently re-routing to Zarr.  This regression
    test creates exactly the ambiguous configuration described in
    the audit (Bug 3) and asserts HDF5 is selected.

    Post-v4.16.1: the directory-routing is gated on the Zarr store
    marker (``zarr.json`` for v3, ``.zarray`` for v2); absent that
    marker the dispatcher falls through to HDF5.
    """
    from lumenairy.io.storage import _detect_backend

    with tempfile.TemporaryDirectory() as tmpdir:
        # Stale directory: name matches a typical HDF5 path but
        # carries no Zarr marker files.
        stale_dir = os.path.join(tmpdir, 'stale_run.h5')
        os.makedirs(stale_dir)
        # The directory exists but is NOT a Zarr store (no
        # zarr.json / .zarray).  Pre-v4.16.1 the bare
        # ``os.path.isdir`` check would silently classify this
        # as Zarr.  Post-fix it must classify as HDF5.
        result = _detect_backend(stale_dir)
        assert result == 'hdf5', (
            f"_detect_backend('{stale_dir}') returned {result!r}; "
            f"expected 'hdf5'.  Directory has no zarr.json / "
            f".zarray marker so the post-v4.16.1 logic should "
            f"reject it as a Zarr store and fall through to HDF5.")


def test_bug3_detect_backend_zarr_marker_routes_to_zarr():
    """A directory with a ``zarr.json`` marker (v3) must dispatch
    to Zarr.

    Pre-v4.16.1 the bare ``os.path.isdir`` routed every directory
    to Zarr -- which got the v3 case right by accident.  Post-fix
    the explicit marker check still routes the v3 case to Zarr.
    Both v3 (``zarr.json``) and v2 (``.zarray``) markers are
    pinned.
    """
    from lumenairy.io.storage import _detect_backend

    with tempfile.TemporaryDirectory() as tmpdir:
        # Zarr v3: zarr.json at the group root.
        v3_store = os.path.join(tmpdir, 'run_v3')
        os.makedirs(v3_store)
        with open(os.path.join(v3_store, 'zarr.json'), 'w') as f:
            f.write('{}')
        assert _detect_backend(v3_store) == 'zarr', (
            "directory with zarr.json (Zarr v3 marker) should "
            "route to 'zarr'")

        # Zarr v2: .zarray at the array root.
        v2_store = os.path.join(tmpdir, 'run_v2')
        os.makedirs(v2_store)
        with open(os.path.join(v2_store, '.zarray'), 'w') as f:
            f.write('{}')
        assert _detect_backend(v2_store) == 'zarr', (
            "directory with .zarray (Zarr v2 marker) should "
            "route to 'zarr'")

        # Bare directory: no marker.  Falls through to HDF5.
        bare = os.path.join(tmpdir, 'bare_dir')
        os.makedirs(bare)
        assert _detect_backend(bare) == 'hdf5', (
            "bare directory without any Zarr marker should "
            "route to 'hdf5'")


def test_bug3_detect_backend_accepts_pathlib_path():
    """``_detect_backend`` must accept ``pathlib.Path`` inputs.

    Pre-v4.16.1: ``path.endswith('.zarr')`` raised
    ``AttributeError`` on a Path (Path objects have no
    ``.endswith``).  Post-fix the ``str(path)`` cast normalises
    the input.
    """
    from lumenairy.io.storage import _detect_backend

    with tempfile.TemporaryDirectory() as tmpdir:
        # .zarr-extension Path.
        zarr_path = Path(tmpdir) / 'out.zarr'
        # Don't actually create it; the dispatcher routes on the
        # extension alone before the isdir check.  The post-fix
        # ``str(path).endswith('.zarr')`` returns True without
        # any AttributeError.
        result = _detect_backend(zarr_path)
        assert result == 'zarr', (
            f"_detect_backend(Path('out.zarr')) returned {result!r}; "
            f"expected 'zarr' via the .zarr extension.")

        # .h5-extension Path.
        h5_path = Path(tmpdir) / 'out.h5'
        result_h5 = _detect_backend(h5_path)
        assert result_h5 == 'hdf5', (
            f"_detect_backend(Path('out.h5')) returned {result_h5!r}; "
            f"expected 'hdf5'.")


# ============================================================================
# Bug 4 -- LM bounds parsing
# ============================================================================
#
# v4.16.2 (audit P3-NEW-F1-8 closure note): scipy's ``least_squares``
# contract is that ``method='lm'`` does NOT accept bounds; passing
# both forces a silent switch to ``method='trf'`` at
# ``optimize/core.py:4190``.  v4.16.2 added a ``UserWarning`` at the
# override point so users notice.  The Bug 4 tests below pass
# ``method='lm'`` together with ``bounds=...`` (the original audit
# scenario was "the LM bound-parsing helper trips on
# ``None``-endpoints"), so they will now emit the override warning
# on every call.  The tests' assertions don't depend on warnings,
# so they continue to pass cleanly; the audit-closure pin for the
# new warning lives in ``test_v4_16_2_agent_b.py``.


def test_bug4_lm_bounds_with_none_lower():
    """Pin: ``bounds=[(None, 1.0), (0.0, None), (-2.0, 2.0)]`` is
    accepted cleanly under ``method='lm'``.

    Pre-v4.16.1: the comprehension ``b[0] if b else -np.inf`` is a
    Python-truthiness trap.  ``b = (None, 1.0)`` is a non-empty
    tuple (always truthy), so ``b[0] = None`` leaked into
    ``np.array(...)``, which produced an object-dtype array that
    scipy's ``least_squares`` rejects with an opaque downstream
    error.  scipy's ``bounds=(lb, ub)`` API spec requires each
    endpoint to be either a finite float OR +/-np.inf -- ``None``
    is NOT in the spec.

    Post-v4.16.1: the explicit
    ``lb = b[0] if (b is not None and b[0] is not None) else -np.inf``
    pattern honours the "no bound on this side" idiom and produces
    a clean float64 array.
    """
    from lumenairy.optimize import (
        CallableMerit,
        DesignParameterization,
        design_optimize,
    )

    template = {
        'params': [0.5, 0.5, 0.0],
        'surfaces': [{'radius': np.inf, 'aperture': 5e-3,
                      'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [],
        'aperture_diameter': 5e-3,
    }
    free_vars: List[Tuple[Any, ...]] = [
        ('params', 0), ('params', 1), ('params', 2)]
    # Mix of None (no bound on that side) and finite bounds.
    bounds = [(None, 1.0), (0.0, None), (-2.0, 2.0)]
    param = DesignParameterization(
        template=template, free_vars=free_vars, bounds=bounds)

    def _q(ctx) -> float:
        x = np.asarray(ctx.x, dtype=np.float64)
        return float(np.sum(x * x))

    merits = [CallableMerit(_q, weight=1.0, name='q')]

    # Pre-v4.16.1: raises TypeError or scipy ValueError about
    # object dtype.  Post-v4.16.1: accepts cleanly.
    result = design_optimize(
        param, merits, wavelength=1.31e-6,
        method='lm', max_iter=20, verbose=False)
    # Just verify the call completed and produced a sensible result.
    assert result is not None
    assert hasattr(result, 'x')
    assert len(result.x) == 3
    assert np.all(np.isfinite(result.x))


def test_bug4_lm_bounds_finite_respected():
    """Sanity: explicit finite bounds are still respected post-fix.

    Build a 1-D problem with merit ``(x - 3)^2`` and a bound
    ``[0, 1]``.  Unconstrained optimum is x*=3; the bound forces
    the solution to x* <= 1.  Post-v4.16.1 the bound endpoint
    parsing handles a finite (lb, ub) tuple identically to
    pre-v4.16.1 (no regression on the common path).
    """
    from lumenairy.optimize import (
        CallableMerit,
        DesignParameterization,
        design_optimize,
    )

    template = {
        'params': [0.5],
        'surfaces': [{'radius': np.inf, 'aperture': 5e-3,
                      'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [],
        'aperture_diameter': 5e-3,
    }
    param = DesignParameterization(
        template=template, free_vars=[('params', 0)],
        bounds=[(0.0, 1.0)])

    def _shifted_quadratic(ctx) -> float:
        x = float(np.asarray(ctx.x, dtype=np.float64)[0])
        return (x - 3.0) ** 2

    merits = [CallableMerit(_shifted_quadratic, weight=1.0,
                            name='shifted')]

    result = design_optimize(
        param, merits, wavelength=1.31e-6,
        method='lm', max_iter=50, verbose=False)
    # Unconstrained optimum is x=3.  Bound [0, 1] forces x <= 1.
    assert result.x[0] <= 1.0 + 1e-6, (
        f"finite upper bound (ub=1.0) violated: x={result.x[0]}.  "
        f"This would suggest the post-fix bounds-parsing path "
        f"silently discarded the explicit finite endpoints.")
    assert result.x[0] >= -1e-6, (
        f"finite lower bound (lb=0.0) violated: x={result.x[0]}.")
    # The solution should be at or near the upper bound (= 1.0).
    assert result.x[0] >= 0.5, (
        f"with merit (x-3)^2 and bounds [0, 1] the optimum should "
        f"push toward x=1; got x={result.x[0]}.")


# ============================================================================
# Meta-test: pin the fix lines (defensive against accidental revert)
# ============================================================================


def test_v4_16_1_fix_lines_present():
    """Defensive pin: each of the four fixes must show its
    signature line in the file at run time.

    A casual revert (someone copy-pastes an old version of one of
    the four files) would re-introduce all four bugs silently.
    The functional regression tests catch the behaviour, but this
    meta-test pins the textual marker so reverts are obvious in
    CI even before the functional tests run.
    """
    repo = Path(__file__).resolve().parents[2]

    # Bug 1 marker: avg divisor in MultiWavelengthMerit.evaluate.
    core_py = (repo / 'lumenairy' / 'optimize' / 'core.py').read_text(
        encoding='cp1252')
    assert 'total / max(len(self.wavelengths), 1)' in core_py, (
        "Bug 1 fix-line missing: "
        "'total / max(len(self.wavelengths), 1)' must appear in "
        "optimize/core.py (MultiWavelengthMerit.evaluate AVG return).")

    # Bug 4 marker: explicit None-aware bounds resolver in LM branch.
    assert '_resolve_bound' in core_py, (
        "Bug 4 fix-line missing: '_resolve_bound' helper must "
        "appear in optimize/core.py (LM bounds parsing).")

    # Bug 2 marker: on-grid pitch in shack_hartmann.
    det_py = (repo / 'lumenairy' / 'analysis' / 'detector.py').read_text(
        encoding='cp1252')
    assert 'pitch_actual = sa_pixels * dx' in det_py, (
        "Bug 2 fix-line missing: 'pitch_actual = sa_pixels * dx' "
        "must appear in analysis/detector.py.")
    assert 'cumsum(sx_safe, axis=1) * pitch_actual' in det_py, (
        "Bug 2 fix-line missing: cumsum(sx_safe, axis=1) * "
        "pitch_actual must appear (instead of * lenslet_pitch).")

    # Bug 3 marker: zarr.json marker check in _detect_backend.
    io_py = (repo / 'lumenairy' / 'io' / 'storage.py').read_text(
        encoding='cp1252')
    assert "'zarr.json'" in io_py and "'.zarray'" in io_py, (
        "Bug 3 fix-line missing: 'zarr.json' and '.zarray' marker "
        "checks must appear in io/storage.py (_detect_backend).")
    assert 'str(path)' in io_py, (
        "Bug 3 fix-line missing: 'str(path)' cast must appear in "
        "io/storage.py (_detect_backend) for pathlib.Path support.")
