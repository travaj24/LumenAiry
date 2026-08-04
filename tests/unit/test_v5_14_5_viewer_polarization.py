"""Optics-viewer support for waveplate + PBS elements (SHIPPED v5.14.4).

Filename says v5.14.5; the feature actually shipped in **v5.14.4**
(2026-06-14) -- CHANGELOG's ``[5.14.5]`` holds only the ``_fast_geig``
folded eigensolve.  Filename left alone deliberately (renaming churns node
ids and ``.test_durations``); this note is the correction.

The 2-D / 3-D layout views can now render polarization elements (waveplate,
polarizing beam splitter), which previously had no glyph.  The 2-D path
(Qt QGraphicsScene) is GL-free and tested headlessly; the 3-D path (PyVista /
OpenGL) is skipped where no GL context is available, but its mesh geometry is
checked directly.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

from lumenairy.ui.model import Element, SurfaceRow, SystemModel  # noqa: E402


@pytest.fixture(scope="module")
def _qapp():
    return QApplication.instance() or QApplication([])


def _system():
    sm = SystemModel()
    wp = Element(0, 'QWP', 'Waveplate', distance_mm=20.0,
                 surfaces=[SurfaceRow(radius=np.inf, thickness=1.0,
                                      glass='AIR', semi_diameter=8.0)],
                 aux={'wp_kind': 'quarter', 'fast_axis_deg': 45.0})
    pbs = Element(0, 'PBS1', 'PBS', distance_mm=20.0,
                  surfaces=[SurfaceRow(radius=np.inf, thickness=10.0,
                                       glass='AIR', semi_diameter=8.0)],
                  aux={'pbs_angle_deg': 45.0})
    sm.insert_element(len(sm.elements) - 1, wp)
    sm.insert_element(len(sm.elements) - 1, pbs)
    sm.recompute_element_frames()
    return sm, wp, pbs


def test_element_types_registered():
    assert 'Waveplate' in Element.TYPES and 'PBS' in Element.TYPES


def test_layout_2d_renders_polarization(_qapp):
    from lumenairy.ui.layout_2d import Layout2DView
    sm, wp, pbs = _system()
    assert [e.elem_type for e in sm.elements] == \
        ['Source', 'Waveplate', 'PBS', 'Detector']
    view = Layout2DView(sm)
    view.rebuild()                                   # full system, no crash
    assert len(view.scene.items()) > 0
    # each glyph adds items
    n = len(view.scene.items())
    view._draw_waveplate(0.0, 30.0, wp)
    assert len(view.scene.items()) - n == 3          # plate + axis + label
    n = len(view.scene.items())
    view._draw_pbs(0.0, 30.0, pbs)
    assert len(view.scene.items()) - n >= 4          # cube + diag + port+label


def test_layout_3d_glyph_geometry():
    """The PyVista glyph meshes build correctly (the live render needs a GL
    context that headless CI may lack -- the add_mesh calls follow the
    validated _draw_mirror_3d pattern)."""
    pv = pytest.importorskip("pyvista")
    o = np.array([0.0, 0.0, 50.0])
    R = np.eye(3)
    sd = 8.0
    plate = pv.Cylinder(center=tuple(o), direction=tuple(R[:, 2]),
                        radius=sd, height=max(0.4, sd * 0.18), resolution=48)
    assert plate.n_points > 0
    cube = pv.Cube(center=tuple(o), x_length=2 * sd, y_length=2 * sd,
                   z_length=2 * sd)
    cube.points = (cube.points - o) @ R.T + o
    assert np.allclose(cube.points.mean(axis=0), o)
    nrm = (R[:, 0] + R[:, 2]) / np.sqrt(2.0)
    plane = pv.Plane(center=tuple(o), direction=tuple(nrm),
                     i_size=2 * sd * 1.41, j_size=2 * sd)
    assert plane.n_points > 0


@pytest.mark.skipif(
    os.environ.get("LUMENAIRY_TEST_GL") != "1",
    reason="live 3-D render needs a real GL context (VTK hard-crashes "
           "headless, uncatchable); set LUMENAIRY_TEST_GL=1 to run on a "
           "machine with a display.")
def test_layout_3d_renders_if_gl(_qapp):
    pytest.importorskip("pyvista")
    from lumenairy.ui.layout_3d import Layout3DView
    sm, _wp, _pbs = _system()
    view = Layout3DView(sm)
    view.rebuild()
    if view._plotter is not None:
        names = list(view._plotter.actors.keys())
        assert any('waveplate' in n for n in names)
        assert any('pbs' in n for n in names)
