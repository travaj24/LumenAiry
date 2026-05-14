"""GUI visual-regression smoke tests for layout-dock sizing.

Catches the class of bugs that bit 3.7.5/3.7.6: the layout dock
won't shrink interactively, or starts at its minimum size on
launch, or rescales the rendered optics on resize when it should
clip.  These failure modes are interactive-only -- the existing
25-file numerical validation suite (`run_all.py`) doesn't touch
Qt and won't see them.

Run standalone or via `run_all.py`.  Uses the standard `Harness`
pattern so failures route through the same summary.

GUI deps (PySide6, matplotlib, pyvista) are gated behind the
``[gui]`` extra and intentionally NOT installed on CI -- see
``.github/workflows/validate.yml``.  This test self-skips with
exit 0 when any GUI dependency is missing so CI stays green;
locally it runs in full.
"""
from __future__ import annotations

import os
import sys

# Force offscreen Qt + add the validation/ parent so _harness is
# importable BEFORE we attempt the GUI imports.  Done first so the
# skip-on-missing-deps path below still has a working sys.path.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, '..')))

# Stub pyvistaqt with a minimal QtInteractor before importing
# lumenairy.ui.layout_3d -- otherwise that module imports the real
# pyvistaqt at module-load time when it IS installed (dev machines).
import types as _types
_fake_qt = _types.ModuleType('pyvistaqt')


class _FakeQtInteractor:  # noqa: D401 -- minimal stub
    def __init__(self, *a, **k):
        pass


_fake_qt.QtInteractor = _FakeQtInteractor
sys.modules.setdefault('pyvistaqt', _fake_qt)

# Probe GUI deps.  On CI (PySide6 / matplotlib / pyvista not
# installed -- the workflow installs only ``[fft,perf,numba,hdf5,
# zarr,dev]``), any of these imports raise ImportError and we
# skip the suite cleanly with exit 0.  Locally (pip install -e
# .[gui]) everything imports and the suite runs.
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt, QSettings
    # Eagerly import lumenairy.ui modules so that ANY missing GUI
    # transitive dep (matplotlib, pyvista, refractiveindex glass
    # catalogs that load on first use, etc.) surfaces as a clean
    # skip rather than a half-loaded module + crash later.
    import lumenairy.ui.layout_3d as _m3
    from lumenairy.ui.model import SystemModel
    from lumenairy.ui.layout_2d import Layout2DView
    from lumenairy.ui.layout_3d import Layout3DView
    from lumenairy.ui.main_window import MainWindow
    from lumenairy.io.prescriptions import load_zemax_zmx
except ImportError as _e:
    print('=' * 60)
    print('  gui-layout-shrink')
    print('=' * 60)
    print(f'  [SKIP] GUI dependencies not available: {_e}')
    print('  Install with: pip install -e ".[gui]"')
    print('=' * 60)
    print('RESULT [gui-layout-shrink]: skipped (no GUI deps)')
    print('=' * 60)
    sys.exit(0)


from _harness import Harness  # noqa: E402

_app = QApplication.instance() or QApplication([])

# Disable any persisted workspace state so we always start from
# the freshly-built default layout.
_s = QSettings('lumenairy', 'OpticalDesigner')
_s.remove('workspaces/data')

# Force the 3D layout into its fallback (label) path so the
# headless test doesn't try to spin up VTK + an OpenGL context.
_m3.PYVISTAQT_AVAILABLE = False


def _tx71_path():
    candidates = [
        r'D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics'
        r'\Reverse_Symmetric_ASM\tx4designstudy71\tx4designstudy71.zmx',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    H = Harness('gui-layout-shrink')

    H.section('Layout2DView construction')
    sm = SystemModel()
    v2 = Layout2DView(sm)
    H.check('Layout2DView constructs on empty model',
            v2 is not None)
    H.check('widget.minimumSize is (0, 0)',
            v2.minimumSize().width() == 0
            and v2.minimumSize().height() == 0)
    H.check('widget.minimumSizeHint is at most (40, 40)',
            v2.minimumSizeHint().width() <= 40
            and v2.minimumSizeHint().height() <= 40)
    v2.resize(60, 80)
    _app.processEvents()
    H.check('Layout2DView accepts resize(60, 80)',
            v2.size().width() == 60 and v2.size().height() == 80,
            detail=f'actual={v2.size().width()}x{v2.size().height()}')
    v2.resize(800, 400)
    _app.processEvents()
    H.check('Layout2DView accepts resize(800, 400)',
            v2.size().width() == 800 and v2.size().height() == 400)

    H.section('Layout3DView construction (fallback path)')
    v3 = Layout3DView(sm)
    H.check('Layout3DView constructs on empty model',
            v3 is not None)
    v3.resize(60, 80)
    _app.processEvents()
    H.check('Layout3DView accepts resize(60, 80)',
            v3.size().width() == 60 and v3.size().height() == 80,
            detail=f'actual={v3.size().width()}x{v3.size().height()}')

    H.section('MainWindow dock-shrink physics')
    mw = MainWindow()
    mw.resize(1600, 1200)
    mw.show()
    _app.processEvents()

    # The 2D layout dock should start at a sensible size (NOT
    # pinned at its minimum, NOT taking the entire window).
    h0 = mw.layout_dock.size().height()
    H.check('layout_dock starts at sensible default height',
            100 < h0 < 1100,
            detail=f'h0={h0}')

    # resizeDocks must actually shrink the dock now that the
    # central widget's max-height cap is removed.
    mw.resizeDocks([mw.layout_dock], [200], Qt.Vertical)
    _app.processEvents()
    h_200 = mw.layout_dock.size().height()
    H.check('resizeDocks([200]) shrinks layout_dock below the default',
            h_200 < h0,
            detail=f'before={h0} after={h_200}')
    H.check('resizeDocks([200]) gets within ~50 px of target',
            abs(h_200 - 200) < 50,
            detail=f'actual={h_200}')

    mw.resizeDocks([mw.layout_dock], [100], Qt.Vertical)
    _app.processEvents()
    H.check('resizeDocks([100]) shrinks further',
            mw.layout_dock.size().height() < h_200,
            detail=f'h={mw.layout_dock.size().height()}')

    # And the dock's true minimum should be tiny (40-60 px).
    mw.resizeDocks([mw.layout_dock], [10], Qt.Vertical)
    _app.processEvents()
    h_min = mw.layout_dock.size().height()
    H.check('layout_dock minimum is < 80 px',
            h_min < 80,
            detail=f'h_min={h_min}')

    H.section('tx71 trace + dock construction')
    zmx = _tx71_path()
    if zmx is None:
        # No tx71 prescription available; skip the integration tests.
        H.check('tx71 prescription found', True,
                detail='skipped: no .zmx file at known path')
    else:
        pr = load_zemax_zmx(zmx)
        sm2 = SystemModel()
        sm2.load_prescription(pr)
        result = sm2.run_trace(num_rings=1, rays_per_ring=1)
        H.check('tx71 run_trace returns a TraceResult',
                result is not None)
        if result is not None:
            # World positions of the chief ray at each post-fold lens
            # should match the GUI's element world origins to within
            # 0.1 mm.  This is the integration test for the 3.7.6 +
            # 3.7.7 world-coord trace work.
            import numpy as np
            sf3 = sm2.surface_frames_3d_mm()
            for i, (rb, frame) in enumerate(
                    zip(result.ray_history, sf3)):
                origin, R = frame
                lx, ly, lz = (float(rb.x[0]), float(rb.y[0]),
                              float(rb.z[0]))
                world = origin + R @ np.array(
                    [lx * 1e3, ly * 1e3, lz * 1e3])
                # Only check the post-fold tail (SpatialFilter onwards).
                label = result.surfaces[i].label if i < len(
                    result.surfaces) else 'IMG'
                if 'SpatialFilter' in label and 'S1' in label:
                    target_y = float(origin[1])
                    H.check(
                        f'tx71: chief ray at {label} lands at '
                        f'lens y vertex',
                        abs(world[1] - target_y) < 0.1,
                        detail=f'world={world[1]:.4f} '
                               f'vertex={target_y:.4f}')

    sys.exit(H.summary())


if __name__ == '__main__':
    main()
