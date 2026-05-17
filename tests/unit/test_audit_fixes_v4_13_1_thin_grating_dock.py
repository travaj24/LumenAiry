"""Pinning tests for the v4.13.1 audit fix P1-D:
``ThinGratingDock._run`` kwargs mismatch.

Audit reference
---------------

``AUDIT_V4_13_0_2026_05_17.md`` P1-D found that the dock's ``_run``
handler called ``la.grating_efficiency_vs_wavelength`` with kwargs
``groove_index=``, ``substrate_index=``, ``profile=``, ``angle=`` --
none of which the function actually accepts.  The required
``n_ridge`` and ``n_superstrate`` parameters were missing entirely.
Every dock click silently turned into a ``TypeError`` swallowed by
the dock's broad ``except Exception`` summary.  The dock was
non-functional.

v4.13.1 fix:

* Extract a pure ``_compute_efficiency_data(inputs)`` helper that
  drives :func:`lumenairy.thin_grating_efficiency_1d` directly and
  returns a result dict.  No Qt state, so the compute path is unit-
  testable without a live ``QApplication`` round-trip.
* Add UI fields for ``n_ridge`` and ``n_superstrate`` (previously
  missing).
* ``_run`` now collects inputs via ``_collect_inputs()``, calls
  ``_compute_efficiency_data()``, and routes the result through
  ``_draw_result()``.

What this test pins
-------------------

1. **Pure helper happy path**: feed a synthetic-but-realistic input
   dict to ``_compute_efficiency_data`` and assert the result dict
   shape, the order axis includes m=0, and the per-wavelength
   efficiency sum is in [0.5, 1.5] (lossless thin phase grating
   sums to ~1 by Parseval).
2. **Wrong-kwargs regression**: confirm the helper does NOT raise
   ``TypeError`` on standard inputs (the exact failure mode the
   pre-fix dock exhibited).
3. **Qt smoke test**: in an offscreen ``QApplication``, construct
   the dock, click-equivalent ``_run()``, and assert the summary
   text box receives a non-error, non-empty status line.

Author: Andrew Traverso -- v4.13.1
"""
from __future__ import annotations

import os

import numpy as np
import pytest


# Force offscreen Qt BEFORE importing PySide6 in any path below.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


# Probe GUI deps: on CI the [gui] extra is not installed and we
# skip cleanly.  Locally everything imports and the suite runs.
try:
    from PySide6.QtWidgets import QApplication  # noqa: F401
    import matplotlib  # noqa: F401
    matplotlib.use('Agg')
    from lumenairy.ui.thin_grating_dock import ThinGratingDock
    from lumenairy.ui.model import SystemModel
    _GUI_OK = True
except ImportError as _e:
    _GUI_OK = False
    _SKIP_REASON = f'GUI deps unavailable: {_e}'


# ============================================================================
# Pure-helper happy path -- no Qt
# ============================================================================

@pytest.mark.skipif(not _GUI_OK,
                    reason=_SKIP_REASON if not _GUI_OK else '')
class TestComputeEfficiencyDataPureHelper:
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
class TestThinGratingDockEndToEnd:
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
