"""v4.15.5 / Agent B -- regression tests for the OPD plotting + validation
helper closures landed in the v4.15.5 patch.

Closes the following findings from
``docs/audits/AUDIT_V4_15_4_2026_05_19.md``:

* P1-NEW-V2-1 -- ``plot_opd_fan`` ``fan_units`` kwarg.  The canonical
  pipeline ``opd_fan_data(...) -> plot_opd_fan(*, units='waves',
  fan_units='waves', wavelength=wl)`` no longer divides by
  ``wavelength`` a second time.  Default ``fan_units='m'`` preserves
  v4.15.4 back-compat.
* P1-NEW-V2-2 -- ``_radial_rms_profile`` uses centered RMS about the
  in-aperture mean, matching ``plot_wavefront`` and the 1-D fan RMS.
* P2-NEW-V2-3 -- ``plot_opd_summary`` docstring explicitly states
  ``opd_2d`` is in metres.
* P2-NEW-V2-4 -- example ``plot_opd_summary_singlet.py`` print agrees
  with the heatmap annotation (centered RMS).
* P2-NEW-F1-3 -- ``_check_2d_scalar_field`` parameterised with
  ``input_kind``; ``richards_wolf_focus`` / ``debye_wolf_psf`` raise
  with "expected 2-D complex pupil" rather than "field".
* P2-NEW-V2-5 -- central-row / column fallback aligned to
  ``(N - 1) // 2`` so the indexing matches the
  ``_radial_rms_profile`` geometric centre for both odd and even N.
* P3-NEW-F1-4 -- ``plot_opd_fan`` docstring specifies "centered RMS".
* P3-NEW-F1-5 -- ``plot_opd_summary`` exposes ``radial_rms_n_bins``
  with ``'auto'`` clamp behaviour for tiny grids.

All tests use matplotlib's ``Agg`` backend so they render headlessly
in CI without an X11 display.

Author: Andrew Traverso -- v4.15.5 / Agent B.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import lumenairy as la  # noqa: E402
from lumenairy.analysis.plotting import (  # noqa: E402
    plot_opd_fan,
    plot_opd_summary,
    _radial_rms_profile,
    _auto_n_bins,
)


# ----------------------------------------------------------------------------
# Synthetic test data
# ----------------------------------------------------------------------------


def _make_defocus_2d(N: int = 64, peak_waves: float = 0.5,
                     wavelength: float = 633e-9):
    """Build a 2-D defocus OPD over a circular aperture (in METRES)."""
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    ap = (X ** 2 + Y ** 2) <= 1.0
    opd_waves = peak_waves * (X ** 2 + Y ** 2)
    opd_m = opd_waves * wavelength
    return opd_m, ap, 1e-6, wavelength


def _make_singlet_surfaces(aperture: float = 10e-3,
                           wavelength: float = 633e-9):
    """Return ``(surfaces, semi_aperture, wavelength)`` for the singlet
    used in the end-to-end ``opd_fan_data`` regression."""
    rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=4e-3, glass='N-BK7',
                          aperture=aperture)
    rx['object_distance'] = 1.0
    surfaces = la.surfaces_from_prescription(rx)
    semi_aperture = rx['aperture_diameter'] / 2.0
    return surfaces, semi_aperture, wavelength


# ----------------------------------------------------------------------------
# B.1 -- plot_opd_fan fan_units kwarg (P1-NEW-V2-1)
# ----------------------------------------------------------------------------


def test_plot_opd_fan_fan_units_waves_no_double_division():
    """B.1.a (P1-NEW-V2-1) -- ``fan_units='waves'`` + ``units='waves'``
    is the identity for the displayed values: the canonical pipeline
    ``opd_fan_data -> plot_opd_fan`` does NOT divide by wavelength a
    second time.
    """
    surfaces, semi_aperture, wavelength = _make_singlet_surfaces()
    py_w, opd_y_w, px_w, opd_x_w = la.opd_fan_data(
        surfaces, wavelength, semi_aperture, field_angle=0.0, n_rays=51)
    fig, (ax_y, ax_x) = plot_opd_fan(
        py_w, opd_y_w, px_w, opd_x_w,
        units='waves', wavelength=wavelength, fan_units='waves',
        show_stats=False)
    y_plotted = ax_y.get_lines()[0].get_ydata()
    x_plotted = ax_x.get_lines()[0].get_ydata()
    # No double-conversion: rendered values must equal opd_fan_data's
    # WAVES output to numerical precision.
    np.testing.assert_allclose(y_plotted, opd_y_w, rtol=1e-12,
                               equal_nan=True)
    np.testing.assert_allclose(x_plotted, opd_x_w, rtol=1e-12,
                               equal_nan=True)
    # Sanity: if the double-division had happened, the values would
    # have been ~6e5x smaller (633 nm).  Pin the order-of-magnitude
    # check so a regression to the pre-v4.15.5 behaviour also fails.
    finite_y = opd_y_w[np.isfinite(opd_y_w)]
    finite_plotted = y_plotted[np.isfinite(y_plotted)]
    if finite_y.size > 0 and np.any(np.abs(finite_y) > 1e-6):
        # If opd_y_w has appreciable magnitude (well above 1 ulp), the
        # plotted ratio must be O(1), not O(1/wavelength).
        ratio = np.abs(finite_plotted).max() / max(
            np.abs(finite_y).max(), 1e-30)
        assert 0.5 <= ratio <= 2.0, (
            f"plot_opd_fan(fan_units='waves', units='waves') ratio "
            f"plotted / input = {ratio:.3e}, expected O(1).  Double-"
            f"conversion regression?")
    plt.close(fig)


def test_plot_opd_fan_fan_units_metres_default_backcompat():
    """B.1.b (P1-NEW-V2-1) -- ``fan_units='m'`` is the default and
    preserves v4.15.4 behaviour: metres-valued input divided by
    ``wavelength`` for display ``units='waves'``.
    """
    n = 51
    wavelength = 633e-9
    p = np.linspace(-1.0, 1.0, n)
    opd_waves = 0.5 * p ** 2
    opd_m = opd_waves * wavelength  # metres
    # Default fan_units='m'.
    fig, (ax_y, ax_x) = plot_opd_fan(
        p, opd_m, p, opd_m,
        units='waves', wavelength=wavelength, show_stats=False)
    y_plotted = ax_y.get_lines()[0].get_ydata()
    # Display = metres / wavelength = the original waves array.
    np.testing.assert_allclose(y_plotted, opd_waves, rtol=1e-12)
    plt.close(fig)


def test_plot_opd_fan_fan_units_invalid_raises():
    """B.1.c (P1-NEW-V2-1) -- bogus ``fan_units`` raises ValueError
    with a useful message.
    """
    p = np.linspace(-1, 1, 11)
    opd = np.zeros_like(p)
    with pytest.raises(ValueError, match=r"fan_units"):
        plot_opd_fan(p, opd, p, opd,
                     units='waves', wavelength=633e-9,
                     fan_units='bogus')


def test_plot_opd_fan_fan_units_waves_requires_wavelength():
    """B.1.d (P1-NEW-V2-1) -- ``fan_units='waves'`` needs a positive
    wavelength to convert input waves to metres before the display-
    units conversion (even if ``units='m'``).
    """
    p = np.linspace(-1, 1, 11)
    opd = np.zeros_like(p)
    with pytest.raises(ValueError, match=r"wavelength"):
        plot_opd_fan(p, opd, p, opd,
                     units='m', fan_units='waves', wavelength=None)


# ----------------------------------------------------------------------------
# B.2 -- centered RMS in _radial_rms_profile (P1-NEW-V2-2)
# ----------------------------------------------------------------------------


def test_radial_rms_profile_centered():
    """B.2.a (P1-NEW-V2-2) -- the per-bin reduction is centered about
    the in-aperture mean.  For a pure-piston OPD (constant in the
    aperture, NaN outside) the centered RMS in every bin is zero;
    the uncentered ``sqrt(mean(opd**2))`` would report the piston
    value squared-rooted.
    """
    N = 64
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    ap = (X ** 2 + Y ** 2) <= 1.0
    piston_value = 0.7e-6  # metres
    opd = np.where(ap, piston_value, np.nan)
    r_centres, rms = _radial_rms_profile(opd, dx=1e-6, dy=1e-6,
                                          aperture=ap, n_bins=8)
    finite = rms[np.isfinite(rms)]
    assert finite.size > 0
    # Every bin should be ~0 (centered RMS of a constant is zero).
    np.testing.assert_allclose(finite, np.zeros_like(finite),
                               atol=1e-15)
    # Sanity: uncentered RMS would have given ~piston_value.  Pin
    # that the test discriminates between the two conventions.
    assert np.max(np.abs(finite)) < 0.5 * abs(piston_value), (
        f"_radial_rms_profile appears to use UNCENTERED RMS; got "
        f"max bin = {np.max(np.abs(finite)):.3g}, piston = "
        f"{piston_value:.3g}.")


def test_radial_rms_profile_matches_heatmap_rms_outer_rim():
    """B.2.b (P1-NEW-V2-2) -- the radial profile's value at the outer
    rim is on the same order of magnitude as the heatmap's centered
    RMS annotation (within a factor of ~3), proving both reductions
    use the same convention.
    """
    opd_m, ap, dx, wl = _make_defocus_2d(N=64, peak_waves=0.5)
    fig, ((ax_hm, ax_rms), (ax_y, ax_x)) = plot_opd_summary(
        opd_m, dx=dx, aperture=ap, units='waves', wavelength=wl)

    import re
    hm_texts = '\n'.join(t.get_text() for t in ax_hm.texts)
    m_rms = re.search(r'RMS:\s*([0-9eE\.\-\+]+)', hm_texts)
    assert m_rms is not None, (
        f"could not parse heatmap RMS annotation: {hm_texts!r}")
    rms_2d = float(m_rms.group(1))

    rms_line = ax_rms.get_lines()[0].get_ydata()
    finite = rms_line[np.isfinite(rms_line)]
    assert finite.size > 0
    rms_outer = float(finite[-1])

    # Centered convention: outer-rim / heatmap ratio is O(1)
    # (typically 1.5-2 for axially-symmetric defocus).  Uncentered
    # would have given ~3.5.  Pin <= 3.0.
    ratio = rms_outer / max(rms_2d, 1e-30)
    assert 0.5 <= ratio <= 3.0, (
        f"radial-RMS outer bin / heatmap RMS ratio = {ratio:.2f}, "
        f"expected ~ 1.5-2 under the centered convention; "
        f"rms_outer = {rms_outer:.4g}, rms_2d = {rms_2d:.4g}.")
    plt.close(fig)


# ----------------------------------------------------------------------------
# B.3 -- plot_opd_summary docstring states metres (P2-NEW-V2-3)
# ----------------------------------------------------------------------------


def test_plot_opd_summary_docstring_states_metres():
    """B.3 (P2-NEW-V2-3) -- the ``opd_2d`` parameter description
    explicitly says "metres" so a v4.15.4 user retracing the docstring
    cannot fall into the units-system trap.
    """
    doc = inspect.getdoc(plot_opd_summary) or ''
    assert 'metres' in doc, (
        "plot_opd_summary docstring must mention 'metres' in the "
        "opd_2d parameter description; v4.15.4 wording 'interpreted "
        "in the units system' was misleading.")
    # Also verify the explicit note about ``units`` controlling only
    # the display conversion.
    assert 'DISPLAY' in doc or 'display' in doc, (
        "plot_opd_summary docstring should clarify that ``units`` "
        "controls only the DISPLAY conversion.")


# ----------------------------------------------------------------------------
# B.5 -- _check_2d_scalar_field input_kind kwarg (P2-NEW-F1-3)
# ----------------------------------------------------------------------------


def test_validation_helper_input_kind_pupil_in_error_message():
    """B.5.a (P2-NEW-F1-3) -- ``richards_wolf_focus`` raises with
    "expected 2-D complex pupil" (NOT "field") when given a wrong-
    shape input, matching its docstring's "pupil function" wording.
    """
    from lumenairy.propagators.vector_diffraction import richards_wolf_focus
    # 3-D input -- ndim != 2 path.
    bad_pupil = np.ones((4, 16, 16), dtype=np.complex128)
    with pytest.raises(ValueError) as excinfo:
        richards_wolf_focus(bad_pupil, wavelength=633e-9,
                            NA=0.5, f=10e-3, dx_pupil=1e-6)
    msg = str(excinfo.value)
    assert 'pupil' in msg, (
        f"richards_wolf_focus rejection message must say 'pupil' "
        f"(input_kind='pupil'); got: {msg!r}")
    assert 'field' not in msg.split('expected 2-D complex')[1].split(' of ')[0], (
        f"richards_wolf_focus rejection should NOT contain 'field' "
        f"after 'expected 2-D complex'; got: {msg!r}")


def test_validation_helper_input_kind_pupil_in_debye_wolf_message():
    """B.5.b (P2-NEW-F1-3) -- ``debye_wolf_psf`` also uses
    ``input_kind='pupil'`` (its signature parallels
    ``richards_wolf_focus``).
    """
    from lumenairy.propagators.vector_diffraction import debye_wolf_psf
    bad_pupil = np.ones((4, 16, 16), dtype=np.complex128)
    with pytest.raises(ValueError) as excinfo:
        debye_wolf_psf(bad_pupil, wavelength=633e-9,
                       NA=0.5, f=10e-3, dx_pupil=1e-6)
    msg = str(excinfo.value)
    assert 'pupil' in msg, (
        f"debye_wolf_psf rejection message must say 'pupil'; got: "
        f"{msg!r}")


def test_validation_helper_input_kind_default_field_backcompat():
    """B.5.c (P2-NEW-F1-3) -- callers that don't pass ``input_kind``
    still see the historic "field" wording, preserving back-compat
    for the ~30 sites that don't use the new kwarg.
    """
    from lumenairy._validation import _check_2d_scalar_field
    bad = np.ones((4, 16, 16), dtype=np.complex128)
    with pytest.raises(ValueError) as excinfo:
        _check_2d_scalar_field(bad, 'fake_fn')
    msg = str(excinfo.value)
    assert 'field' in msg, (
        f"_check_2d_scalar_field default input_kind='field' should "
        f"include 'field' in the rejection message; got: {msg!r}")


# ----------------------------------------------------------------------------
# B.6 -- even-N central-row/col fallback (P2-NEW-V2-5)
# ----------------------------------------------------------------------------


def test_plot_opd_summary_even_n_central_index_aligned_with_radial():
    """B.6 (P2-NEW-V2-5) -- for even N the central-row / column
    fallback uses ``(N - 1) // 2`` so the extracted slice index lines
    up with the geometric centre that ``_radial_rms_profile`` uses
    (``(N - 1) / 2.0``).  Pre-v4.15.5 used ``N // 2``, which was
    one half-pixel right of centre for even N.
    """
    # Build an OPD with a sentinel value at (N-1)//2 (the new centre
    # index) so we can verify the extraction picks it up.
    N = 32  # even N
    opd_m = np.full((N, N), np.nan, dtype=float)
    # Pre-v4.15.5 the centre col was Nx // 2 = 16; v4.15.5 makes it
    # (Nx - 1) // 2 = 15.  Place a sentinel at col 15 only.
    col_new_centre = (N - 1) // 2  # = 15
    col_old_centre = N // 2  # = 16
    sentinel = 0.123e-6
    # Fill BOTH columns with NaN except for col_new_centre.  The
    # extraction must read the sentinel, which is only at the new
    # centre.
    opd_m[:, col_new_centre] = sentinel
    # NaN everywhere else (already from full).
    fig, ((ax_hm, ax_rms), (ax_y, ax_x)) = plot_opd_summary(
        opd_m, dx=1e-6, units='m')
    # The y-fan extraction reads opd_2d[:, col_centre]; its plotted
    # values must be the sentinel (not NaN).
    y_plotted = ax_y.get_lines()[0].get_ydata()
    # Filter finite values.
    finite = y_plotted[np.isfinite(y_plotted)]
    # If the extraction picked col_new_centre (== 15) we'd see the
    # sentinel.  If it picked col_old_centre (== 16) we'd see all
    # NaN.  Pin the new convention.
    assert finite.size > 0, (
        f"plot_opd_summary even-N central-col fallback extracted "
        f"all NaN, expected the sentinel at col {col_new_centre}.  "
        f"Did the index regress to N // 2?")
    np.testing.assert_allclose(finite[0], sentinel, rtol=1e-12)
    plt.close(fig)


# ----------------------------------------------------------------------------
# B.7 -- plot_opd_fan docstring specifies centered RMS (P3-NEW-F1-4)
# ----------------------------------------------------------------------------


def test_plot_opd_fan_docstring_specifies_centered_rms():
    """B.7 (P3-NEW-F1-4) -- the ``plot_opd_fan`` docstring states
    that RMS is the centered RMS about the in-aperture mean (so users
    don't have to reverse-engineer the convention).
    """
    doc = inspect.getdoc(plot_opd_fan) or ''
    assert 'centered' in doc.lower(), (
        "plot_opd_fan docstring must specify the centered-RMS "
        "convention; v4.15.4 said only 'RMS' which was ambiguous.")
    # The docstring should also reference the in-aperture mean.
    assert 'mean' in doc.lower(), (
        "plot_opd_fan docstring should reference the in-aperture "
        "mean to fully disambiguate the centered convention.")


# ----------------------------------------------------------------------------
# B.8 -- n_bins auto-clamp for tiny grids (P3-NEW-F1-5)
# ----------------------------------------------------------------------------


def test_n_bins_kwarg_auto_clamps_for_tiny_grids():
    """B.8.a (P3-NEW-F1-5) -- on a 16x16 grid, ``radial_rms_n_bins=
    'auto'`` produces fewer than 32 bins so the rendered line has
    no fragmented NaN segments.
    """
    opd_m, ap, dx, wl = _make_defocus_2d(N=16, peak_waves=0.5)
    fig, ((ax_hm, ax_rms), (ax_y, ax_x)) = plot_opd_summary(
        opd_m, dx=dx, aperture=ap,
        units='waves', wavelength=wl,
        radial_rms_n_bins='auto')
    line = ax_rms.get_lines()[0]
    y_data = line.get_ydata()
    # Auto-clamp pin: with N=16 and a circular aperture, the number
    # of bins should be less than 32 (the v4.15.4 fixed value).
    assert len(y_data) < 32, (
        f"plot_opd_summary radial_rms_n_bins='auto' should clamp "
        f"below 32 on a 16x16 grid; got {len(y_data)} bins.")
    # And every (or nearly every) bin should be finite -- no
    # fragmented NaN runs from empty annuli.
    finite_count = int(np.sum(np.isfinite(y_data)))
    assert finite_count >= len(y_data) - 1, (
        f"plot_opd_summary radial_rms_n_bins='auto' produced "
        f"{len(y_data) - finite_count} empty bins on a 16x16 grid; "
        f"the auto-clamp should pick a bin count where every bin "
        f"has at least one in-aperture pixel.")
    plt.close(fig)


def test_n_bins_kwarg_explicit_int_overrides_auto():
    """B.8.b (P3-NEW-F1-5) -- passing an explicit ``radial_rms_n_bins
    = N`` bypasses the auto-clamp and uses exactly N bins.
    """
    opd_m, ap, dx, wl = _make_defocus_2d(N=64, peak_waves=0.5)
    fig, ((ax_hm, ax_rms), (ax_y, ax_x)) = plot_opd_summary(
        opd_m, dx=dx, aperture=ap,
        units='waves', wavelength=wl,
        radial_rms_n_bins=16)
    y_data = ax_rms.get_lines()[0].get_ydata()
    assert len(y_data) == 16, (
        f"plot_opd_summary radial_rms_n_bins=16 should use exactly "
        f"16 bins; got {len(y_data)}.")
    plt.close(fig)


def test_n_bins_kwarg_invalid_raises():
    """B.8.c (P3-NEW-F1-5) -- non-positive integer for the kwarg
    raises ValueError.
    """
    opd_m, ap, dx, wl = _make_defocus_2d(N=32)
    with pytest.raises(ValueError, match=r"radial_rms_n_bins"):
        plot_opd_summary(
            opd_m, dx=dx, aperture=ap,
            units='waves', wavelength=wl,
            radial_rms_n_bins=0)


def test_auto_n_bins_helper_returns_at_least_one():
    """B.8.d (P3-NEW-F1-5) -- ``_auto_n_bins`` helper degrades to a
    sensible minimum (>= 1) for degenerate inputs.
    """
    assert _auto_n_bins(0) >= 1
    assert _auto_n_bins(1) >= 1
    assert _auto_n_bins(4) >= 1
    # Sanity on the ceiling clamp.
    assert _auto_n_bins(10_000, ceiling=32) <= 32


# ----------------------------------------------------------------------------
# Cross-cutting -- import sanity + version
# ----------------------------------------------------------------------------


def test_lumenairy_imports_clean():
    """v4.15.5 -- ``import lumenairy`` is still side-effect-clean."""
    import importlib
    import lumenairy
    importlib.reload(lumenairy)
    assert hasattr(lumenairy, 'plot_opd_fan')
    assert hasattr(lumenairy, 'plot_opd_summary')
    assert hasattr(lumenairy, 'opd_fan_data')
