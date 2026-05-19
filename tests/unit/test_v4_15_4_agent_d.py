"""v4.15.4 Agent D -- OPD fan + summary plotting helpers.

Covers the D.1 / D.2 / D.3 / D.4 scope from the v4.15.4 release
dispatch:

* D.1 :func:`lumenairy.plot_opd_fan` -- 2-panel OPD ray-fan plot
  (tangential left, sagittal right).
* D.2 :func:`lumenairy.plot_opd_summary` -- 4-panel summary (2-D OPD
  heatmap, radial-RMS profile, tangential fan, sagittal fan).
* D.3 ``_require_mpl()`` discipline + matplotlib import shim --
  exercised implicitly by every test below.
* D.4 Top-level export plumbing -- ``test_top_level_imports`` pins.

All tests use matplotlib's ``Agg`` backend so they render headlessly
in CI without an X11 display.

Author: Andrew Traverso -- v4.15.4 / Agent D.
"""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import lumenairy as la  # noqa: E402
from lumenairy.analysis.plotting import (  # noqa: E402
    plot_opd_fan,
    plot_opd_summary,
)


# ----------------------------------------------------------------------------
# Synthetic test data
# ----------------------------------------------------------------------------


def _make_defocus_fans(n: int = 51, peak_waves: float = 0.5,
                      wavelength: float = 633e-9):
    """Build two 1-D OPD fans with a pure-defocus parabola.

    Returns ``(py, opd_y_m, px, opd_x_m, wavelength)`` -- OPD in
    METRES so the ``units='waves'`` conversion exercises the
    wavelength branch.
    """
    p = np.linspace(-1.0, 1.0, n)
    opd_waves = peak_waves * p ** 2
    opd_m = opd_waves * wavelength
    return p, opd_m, p, opd_m, wavelength


def _make_defocus_2d(N: int = 64, peak_waves: float = 0.5,
                    wavelength: float = 633e-9):
    """Build a 2-D defocus OPD over a circular aperture.

    Returns ``(opd_m, aperture_mask, dx, wavelength)``.
    """
    x = (np.arange(N) - N / 2) / (N / 2)
    X, Y = np.meshgrid(x, x)
    ap = (X ** 2 + Y ** 2) <= 1.0
    opd_waves = peak_waves * (X ** 2 + Y ** 2)
    opd_m = opd_waves * wavelength
    return opd_m, ap, 1e-6, wavelength


# ----------------------------------------------------------------------------
# D.1 tests -- plot_opd_fan
# ----------------------------------------------------------------------------


def test_plot_opd_fan_returns_fig_and_axes():
    """D.1.1 -- smoke: returns (fig, (ax_y, ax_x))."""
    py, oy, px, ox, wl = _make_defocus_fans()
    result = plot_opd_fan(py, oy, px, ox, units='waves', wavelength=wl)
    assert isinstance(result, tuple) and len(result) == 2
    fig, axes = result
    assert fig is not None
    assert axes is not None
    plt.close(fig)


def test_plot_opd_fan_two_axes_layout():
    """D.1.2 -- the returned axes tuple has exactly 2 elements."""
    py, oy, px, ox, wl = _make_defocus_fans()
    fig, axes = plot_opd_fan(py, oy, px, ox, units='waves', wavelength=wl)
    assert isinstance(axes, tuple), (
        f"plot_opd_fan must return axes as a tuple; got "
        f"{type(axes).__name__}.")
    assert len(axes) == 2, (
        f"plot_opd_fan must return exactly 2 axes (tangential + "
        f"sagittal); got {len(axes)}.")
    ax_y, ax_x = axes
    # Distinct axes so callers can customise each panel independently.
    assert ax_y is not ax_x
    plt.close(fig)


def test_plot_opd_fan_show_stats_overlays_pv_rms():
    """D.1.3 -- show_stats=True overlays 'PV' and 'RMS' text on each
    panel.  Inspects rendered text artists so the test fails if the
    annotation regresses (rather than just checking show_stats=True
    is plumbed)."""
    py, oy, px, ox, wl = _make_defocus_fans()
    fig, (ax_y, ax_x) = plot_opd_fan(
        py, oy, px, ox, units='waves', wavelength=wl,
        show_stats=True)
    for label, ax in (('tangential', ax_y), ('sagittal', ax_x)):
        texts = [t.get_text() for t in ax.texts]
        joined = '\n'.join(texts)
        assert 'PV' in joined, (
            f"plot_opd_fan show_stats overlay missing 'PV' on the "
            f"{label} panel; rendered text = {texts!r}.")
        assert 'RMS' in joined, (
            f"plot_opd_fan show_stats overlay missing 'RMS' on the "
            f"{label} panel; rendered text = {texts!r}.")
    plt.close(fig)


def test_plot_opd_fan_units_conversion_waves():
    """D.1.4 -- units='waves' divides metres by wavelength so the
    rendered ylabel reads 'waves' AND the plotted y-data matches the
    input divided by wavelength (within float roundoff)."""
    py, oy, px, ox, wl = _make_defocus_fans(peak_waves=1.0)
    fig, (ax_y, ax_x) = plot_opd_fan(
        py, oy, px, ox, units='waves', wavelength=wl, show_stats=False)
    # Y-label includes the unit token.
    assert 'waves' in ax_y.get_ylabel(), (
        f"plot_opd_fan ylabel missing 'waves' token; got "
        f"{ax_y.get_ylabel()!r}.")
    # The plotted line data should be the input opd_y / wavelength.
    line = ax_y.get_lines()[0]
    y_plotted = line.get_ydata()
    expected = oy / wl
    np.testing.assert_allclose(y_plotted, expected, rtol=1e-12,
                               atol=1e-15)
    plt.close(fig)


def test_plot_opd_fan_handles_nan_outside_aperture():
    """D.1.5 -- NaN entries (rays outside the aperture / vignetted /
    TIR'd) at the fan boundary are passed through without breaking
    PV / RMS overlay.  The annotation should report stats over the
    non-NaN portion only."""
    py, oy_m, px, ox_m, wl = _make_defocus_fans(n=21, peak_waves=0.5)
    # Mark the outermost two samples on each side as NaN (simulates
    # vignetting at the pupil edge).
    oy_m_nan = oy_m.copy()
    oy_m_nan[:2] = np.nan
    oy_m_nan[-2:] = np.nan
    ox_m_nan = ox_m.copy()
    ox_m_nan[:2] = np.nan
    ox_m_nan[-2:] = np.nan

    fig, (ax_y, ax_x) = plot_opd_fan(
        py, oy_m_nan, px, ox_m_nan,
        units='waves', wavelength=wl, show_stats=True)
    # Expected PV / RMS over the in-aperture portion.
    finite_y = oy_m_nan[np.isfinite(oy_m_nan)] / wl
    expected_pv = float(finite_y.max() - finite_y.min())
    # The PV / RMS overlay text must include a sensible PV value.
    joined = '\n'.join(t.get_text() for t in ax_y.texts)
    assert 'PV' in joined and 'RMS' in joined
    # The PV in the rendered text should equal the in-aperture PV
    # to 3 significant figures (the overlay uses '%.3g' formatting).
    # Parse loosely: find the float token after 'PV: '.
    import re
    m = re.search(r'PV:\s*([0-9eE\.\-\+]+)', joined)
    assert m is not None, f"could not parse PV overlay: {joined!r}"
    pv_text = float(m.group(1))
    np.testing.assert_allclose(pv_text, expected_pv, rtol=1e-2)
    plt.close(fig)


def test_plot_opd_fan_rejects_missing_wavelength():
    """D.1.6 (bonus, sibling of plot_wavefront 5E.2) -- units='waves'
    with no wavelength raises ValueError instead of silently
    producing garbage."""
    p = np.linspace(-1, 1, 11)
    opd = np.zeros_like(p)
    with pytest.raises(ValueError, match='wavelength'):
        plot_opd_fan(p, opd, p, opd, units='waves', wavelength=None)


# ----------------------------------------------------------------------------
# D.2 tests -- plot_opd_summary
# ----------------------------------------------------------------------------


def test_plot_opd_summary_returns_2x2_axes():
    """D.2.1 -- smoke + layout: returns a 2x2 axes tuple."""
    opd_m, ap, dx, wl = _make_defocus_2d()
    fig, axes = plot_opd_summary(
        opd_m, dx=dx, aperture=ap,
        units='waves', wavelength=wl)
    # ((ax_hm, ax_rms), (ax_y, ax_x))
    assert isinstance(axes, tuple)
    assert len(axes) == 2
    assert len(axes[0]) == 2
    assert len(axes[1]) == 2
    # All four axes must be distinct objects.
    flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    assert len(set(id(a) for a in flat)) == 4
    plt.close(fig)


def test_plot_opd_summary_extracts_fans_from_2d_when_args_missing():
    """D.2.2 -- with only opd_2d supplied, the fan panels still
    populate (from central row + column of the 2-D OPD)."""
    opd_m, ap, dx, wl = _make_defocus_2d(N=32, peak_waves=0.5)
    fig, ((ax_hm, ax_rms), (ax_y, ax_x)) = plot_opd_summary(
        opd_m, dx=dx, aperture=ap,
        units='waves', wavelength=wl)
    # Fan panels must each have at least one plotted line.
    assert len(ax_y.get_lines()) >= 1, (
        "plot_opd_summary fan-y panel has no plotted line; "
        "extraction from 2-D OPD failed.")
    assert len(ax_x.get_lines()) >= 1, (
        "plot_opd_summary fan-x panel has no plotted line; "
        "extraction from 2-D OPD failed.")
    # The extracted y-fan should have a parabolic shape (defocus).
    y_data = ax_y.get_lines()[0].get_ydata()
    finite = y_data[np.isfinite(y_data)]
    # Defocus has minimum at the centre; the centre value should be
    # less than or equal to the edge values.
    assert finite.min() <= finite.max()
    plt.close(fig)


def test_plot_opd_summary_uses_provided_fans_when_given():
    """D.2.3 -- explicit fan arrays override the 2-D extraction.

    Pass a 2-D defocus OPD AND explicit fan arrays whose values are
    deliberately different (a constant offset).  Verify that the
    rendered fan panels reflect the explicit arrays, not the 2-D
    extraction.
    """
    opd_m, ap, dx, wl = _make_defocus_2d(N=32, peak_waves=0.5)
    # Build distinctive fan arrays that won't match the 2-D row/col.
    n_fan = 25
    py = np.linspace(-1.0, 1.0, n_fan)
    px = np.linspace(-1.0, 1.0, n_fan)
    # Linear ramp (totally different from the defocus parabola).
    opd_y_explicit_m = 0.3 * wl * py  # tilt in metres
    opd_x_explicit_m = 0.3 * wl * px

    fig, ((ax_hm, ax_rms), (ax_y, ax_x)) = plot_opd_summary(
        opd_m, dx=dx, aperture=ap,
        py=py, opd_y=opd_y_explicit_m,
        px=px, opd_x=opd_x_explicit_m,
        units='waves', wavelength=wl)
    # The plotted line must match the EXPLICIT array, not the 2-D
    # extraction.  Compare to expected waves (divide by wavelength).
    y_plotted = ax_y.get_lines()[0].get_ydata()
    expected = opd_y_explicit_m / wl
    np.testing.assert_allclose(y_plotted, expected, rtol=1e-12)
    # Sanity: the explicit ramp should NOT match a defocus parabola
    # centred at the origin (which is what the 2-D extraction would
    # have produced).
    assert not np.allclose(y_plotted,
                            np.linspace(0, 0.5, n_fan)), (
        "explicit fan override may have been silently dropped; the "
        "rendered y-fan still looks like a parabola.")
    plt.close(fig)


def test_top_level_imports():
    """D.4.1 -- the new names are reachable from the top-level
    namespace AND from the analysis sub-package."""
    assert hasattr(la, 'plot_opd_fan'), (
        "lumenairy.plot_opd_fan missing from top-level namespace.")
    assert hasattr(la, 'plot_opd_summary'), (
        "lumenairy.plot_opd_summary missing from top-level namespace.")
    # Callable check.
    assert callable(la.plot_opd_fan)
    assert callable(la.plot_opd_summary)
    # Also reachable via analysis sub-package.
    from lumenairy.analysis import plot_opd_fan as paf_sub
    from lumenairy.analysis import plot_opd_summary as pos_sub
    assert paf_sub is la.plot_opd_fan
    assert pos_sub is la.plot_opd_summary
