"""Audit S4-3 (v5.24.2 exhaustive audit) [P1][physics]: the wave-optics
dock's default ``lens_model='asm'`` per-surface phase screen called the
rotationally-symmetric ``surface_sag_general(h_sq, radius, conic)`` and
gated on ``np.isfinite(ts.radius)`` alone.  A cylindrical surface
(``radius = inf`` in x, finite ``radius_y``) therefore received NO phase
at all -- it focused to a POINT (i.e. did nothing) instead of a line --
and a biconic/toroidal surface silently collapsed to its x-base conic,
even though ``layout_2d.py:429`` DREW the full biconic sag.

Fix under test (lumenairy/ui/waveoptics_dock.py, ``_run_impl`` phase
screen): route the screen through ``surface_sag_biconic`` and widen the
guard to ``np.isfinite(radius) or np.isfinite(radius_y)`` so a y-only
cylinder still gets power.  ``surface_sag_biconic`` reduces to
``surface_sag_general`` when ``radius_y is None``, so rotationally
symmetric surfaces stay byte-identical.

Independent oracle: run the real worker end-to-end on a single refracting
surface, grab the field saved immediately AFTER the phase screen (the
per-surface plane, thickness=0 so no propagation intervenes), and compare
the applied phase against the paraxial thin-lens power
``phi(Y) = -k (n2-n1) Y^2 / (2 R)`` -- a formula derived independently of
``surface_sag_biconic``'s internals.

Pre-fix the cylinder run applies zero phase, so the recovered curvature is
~0 and ``test_cylinder_surface_gets_y_only_power`` fails; post-fix it
matches the thin-lens oracle and the x-axis stays flat (the astigmatic
signature).
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
# Force offscreen Qt BEFORE importing PySide6 in any path below.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

# Probe GUI deps: on CI the [gui] extra is not installed and we skip the
# worker tests cleanly (same guard the sibling dock tests use).  Locally
# everything imports and runs.
try:
    import matplotlib  # noqa: F401
    from PySide6.QtWidgets import QApplication
    matplotlib.use('Agg')
    from lumenairy.ui.waveoptics_dock import WaveOpticsWorker
    _GUI_OK = True
    _SKIP_REASON = ''
except ImportError as _e:
    _GUI_OK = False
    _SKIP_REASON = f'GUI deps unavailable: {_e}'

# Deterministic constant-index "glass" so n2 does not depend on the
# Sellmeier database (see glass.get_glass_index docstring example).
_GLASS_KEY = 'LUMTEST_S4_3_N1p5'
_N2 = 1.5
_WV = 550e-9
_N = 128
_DX = 15e-6
_EPD = 1.5e-3


class _Surf:
    """Minimal trace-Surface stand-in carrying the biconic fields the
    dock's phase-screen block reads (radius/conic + radius_y/conic_y +
    aspheric_coeffs[_y]) plus the flags the pre-filter checks."""

    is_mirror = False
    is_coordbrk = False
    tilt_x_deg = 0.0
    tilt_y_deg = 0.0
    conic = 0.0
    conic_y = None
    aspheric_coeffs = None
    aspheric_coeffs_y = None
    semi_diameter = float('inf')
    thickness = 0.0          # 0 -> the per-surface plane is saved with
    #                          NO propagation applied after the screen.
    glass_before = 'air'
    glass_after = _GLASS_KEY

    def __init__(self, label, radius, radius_y=None):
        self.label = label
        self.radius = radius
        self.radius_y = radius_y


class _StubModel:
    """Just enough SystemModel surface for _snapshot_model + _run_impl."""

    wavelength_m = _WV
    epd_m = _EPD
    elements = [object()]
    source = None            # -> circular top-hat plane wave of dia EPD.

    def __init__(self, surf):
        self._surf = surf

    def build_trace_surfaces(self):
        return [self._surf]


def _run_worker(surf):
    """Run one worker synchronously (direct-connection signals, no event
    loop) and return its finished payload dict."""
    QApplication.instance() or QApplication([])
    from lumenairy.glass import GLASS_REGISTRY
    GLASS_REGISTRY[_GLASS_KEY] = lambda wl: _N2
    try:
        cfg = {'N': _N, 'dx_m': _DX, 'method': 'asm'}
        worker = WaveOpticsWorker(_StubModel(surf), cfg)
        got = []
        worker.finished_result.connect(got.append)
        worker.run()
        assert len(got) == 1, got
        assert 'error' not in got[0], got[0].get('error')
        return got[0]
    finally:
        GLASS_REGISTRY.pop(_GLASS_KEY, None)


def _post_surface_field(payload, label):
    """Return (field, dx) for the plane saved right after the named
    surface's phase screen."""
    planes = payload['planes']
    match = [p for p in planes if p['label'] == label]
    assert match, [p['label'] for p in planes]
    p = match[-1]
    return p['field'], p['dx']


def _fit_quadratic_coeff(field, dx, axis):
    """Fit the applied phase along the central line of ``axis`` ('col'
    = x=0 column, varying Y; 'row' = y=0 row, varying X) to
    ``phi = a * s^2 + b`` over the illuminated aperture and return
    ``(a, phase_span)``."""
    n = field.shape[0]
    c = n // 2
    coord = (np.arange(n) - n / 2) * dx
    if axis == 'col':          # fixed x = coord[c] = 0, vary Y (rows)
        line = field[:, c]
    else:                      # fixed y = 0, vary X (cols)
        line = field[c, :]
    inside = np.abs(line) > 0.5
    s = coord[inside]
    phi = np.angle(line[inside])
    # Aperture is small (Y/R ~ 1e-3): the exact-conic phase is < pi, so
    # np.angle is already unwrapped -- no unwrap needed.
    a, _b = np.polyfit(s ** 2, phi, 1)
    return a, float(phi.max() - phi.min())


@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
def test_cylinder_surface_gets_y_only_power():
    """A y-only cylinder (radius=inf, finite radius_y) must imprint the
    paraxial thin-lens phase along y and stay flat along x.  Pre-fix the
    infinite-radius guard skipped the screen entirely (a=0), so this
    fails; post-fix it matches the independent thin-lens oracle."""
    R_y = 1.0
    payload = _run_worker(_Surf('CYL', radius=float('inf'), radius_y=R_y))
    field, dx = _post_surface_field(payload, 'CYL')

    k = 2 * np.pi / _WV
    a_expected = -k * (_N2 - 1.0) / (2.0 * R_y)   # phi = a * Y^2

    a_y, span_y = _fit_quadratic_coeff(field, dx, 'col')
    a_x, span_x = _fit_quadratic_coeff(field, dx, 'row')

    # y-axis power matches the thin-lens oracle to well under 1%
    # (paraxial-vs-conic error here is O((Y/R)^2) ~ 1e-7).
    assert a_expected != 0.0
    assert abs(a_y - a_expected) / abs(a_expected) < 0.03, (a_y, a_expected)
    # Astigmatic signature: the x-axis carries no power (flat phase),
    # i.e. the surface is NOT rotationally symmetric.
    assert span_x < 1e-3 * span_y, (span_x, span_y)


@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
def test_rotationally_symmetric_surface_unchanged():
    """A spherical surface (finite radius, radius_y=None) routes through
    the biconic->general reduction and must imprint EQUAL x and y power
    (both matching the thin-lens oracle) -- guards that the refactor left
    the rotationally-symmetric path byte-consistent."""
    R = 1.0
    payload = _run_worker(_Surf('SPH', radius=R, radius_y=None))
    field, dx = _post_surface_field(payload, 'SPH')

    k = 2 * np.pi / _WV
    a_expected = -k * (_N2 - 1.0) / (2.0 * R)

    a_y, _ = _fit_quadratic_coeff(field, dx, 'col')
    a_x, _ = _fit_quadratic_coeff(field, dx, 'row')

    assert abs(a_y - a_expected) / abs(a_expected) < 0.03, (a_y, a_expected)
    assert abs(a_x - a_expected) / abs(a_expected) < 0.03, (a_x, a_expected)
    # x and y powers agree (rotationally symmetric).
    assert abs(a_x - a_y) / abs(a_expected) < 0.03, (a_x, a_y)
