"""Audit S4-2 (v5.24.2 exhaustive audit) [P1][seam]: the GUI wave
optimizer built its per-variable *bounds* from
``SystemModel.get_variable_values()`` -- which returns lengths in
MILLIMETRES -- while the starting point ``x0`` that
``DesignParameterization.initial_values()`` reads comes from the
prescription template ``to_prescription()`` in METRES (model.py
converts mm -> m).  A singlet R = +-50 mm, gap = 20 mm therefore gave
``x0 = [0.05, -0.05, 0.02]`` (m) against boxes ``[(25, 100),
(-100, -25), (10, 40)]`` (mm): x0 sits outside EVERY box, so scipy
clips the start to a nonsense (e.g. 25-metre-radius) design and the
flagship "Wave Optimize" button is broken for all length variables.

Fix under test (lumenairy/ui/optimizer_dock.py, ``_start_wave_optimize``):
build each bound's centre ``val`` from the metre-unit template ``pres``
at the SAME ``path`` used for the free variable, instead of from
``get_variable_values()`` (mm).

Independent oracle: after the (monkeypatched) worker captures the real
``DesignParameterization`` the dock hands it, x0 read from the template
must lie INSIDE its own bounds box for every variable -- the exact
precondition scipy's L-BFGS-B needs to avoid clipping the start.  This
is a property of the parameterization (x0 in [lo, hi]), not a re-run of
the bound formula, so it is not a tautology.  Pre-fix the metre x0 sits
outside the mm boxes and the assertion fails; post-fix it is bracketed.
"""
import os
import types

os.environ.setdefault('OMP_NUM_THREADS', '4')
# Force offscreen Qt BEFORE importing PySide6 in any path below.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest

# Probe GUI deps: on CI the [gui] extra is not installed and we skip
# cleanly (same guard the sibling dock tests use).  Locally everything
# imports and runs.
try:
    from lumenairy.ui import optimizer_dock as _dock_mod
    from lumenairy.ui.model import Element, SourceDefinition, SurfaceRow, SystemModel
    _GUI_OK = True
    _SKIP_REASON = ''
except ImportError as _e:
    _GUI_OK = False
    _SKIP_REASON = f'GUI deps unavailable: {_e}'


# --- geometry (all lengths in mm, as SurfaceRow / Element store them) ---
_R1_MM = 50.0     # front radius  -> +0.05 m in the metre-unit template
_R2_MM = -50.0    # rear radius   -> -0.05 m
_GAP_MM = 20.0    # air gap before the 2nd singlet -> +0.02 m


def _build_singlet_pair_model():
    """A source + two singlets + detector.  Free vars: both radii of the
    first singlet (surface fields) and the air gap preceding the second
    singlet (a ``distance`` -> ``thicknesses`` mapping).  All three are
    length variables, so all three expose the mm/metre unit bug."""
    sm = SystemModel()
    singlet_a = Element(
        1, 'A', 'Singlet', distance_mm=30.0, surfaces=[
            SurfaceRow(radius=_R1_MM, thickness=5.0, glass='N-BK7',
                       semi_diameter=10.0),
            SurfaceRow(radius=_R2_MM, thickness=0.0, glass='',
                       semi_diameter=10.0),
        ])
    singlet_b = Element(
        2, 'B', 'Singlet', distance_mm=_GAP_MM, surfaces=[
            SurfaceRow(radius=80.0, thickness=4.0, glass='N-BK7',
                       semi_diameter=10.0),
            SurfaceRow(radius=-80.0, thickness=0.0, glass='',
                       semi_diameter=10.0),
        ])
    sm.elements = [
        Element(0, 'Source', 'Source', distance_mm=0.0,
                source=SourceDefinition()),
        singlet_a,
        singlet_b,
        Element(3, 'Detector', 'Detector', distance_mm=100.0),
    ]
    # (elem_idx into sm.elements, surf_idx, field)
    sm.opt_variables = [
        (1, 0, 'radius'),     # A front radius
        (1, 1, 'radius'),     # A rear radius
        (2, 0, 'distance'),   # air gap Source-side of singlet B
    ]
    sm._invalidate()
    return sm


class _CaptureWorker:
    """Stand-in for WaveOptimizeWorker: record the DesignParameterization
    the dock constructs and do NOT spin up a QThread."""

    last = None

    def __init__(self, param, *args, **kwargs):
        _CaptureWorker.last = param
        self.finished_result = types.SimpleNamespace(connect=lambda *_a: None)
        self.fine_progress = types.SimpleNamespace(connect=lambda *_a: None)

    def start(self):
        pass


def _make_dock_stub(sm):
    """A minimal stand-in carrying only the attributes
    ``_start_wave_optimize`` touches, so we can drive the REAL method
    without building the whole Qt widget tree (the same
    real-method-on-a-stub pattern the v5.4.2 run_trace test uses)."""
    noop = types.SimpleNamespace(
        setEnabled=lambda *_a: None, setVisible=lambda *_a: None,
        setRange=lambda *_a: None, setValue=lambda *_a: None,
        setFormat=lambda *_a: None)
    return types.SimpleNamespace(
        sm=sm,
        log=types.SimpleNamespace(append=lambda *_a: None),
        spin_target=types.SimpleNamespace(value=lambda: 100.0),
        # geo index 1 -> FocalLengthMerit (a non-empty merit list so the
        # method reaches the worker construction); wave index 0 -> none.
        combo_merit_geo=types.SimpleNamespace(currentIndex=lambda: 1),
        combo_merit_wave=types.SimpleNamespace(currentIndex=lambda: 0),
        chk_jax=types.SimpleNamespace(isChecked=lambda: False),
        spin_iter=types.SimpleNamespace(value=lambda: 10),
        btn_optimize=noop, btn_global=noop, btn_wave=noop, btn_stop=noop,
        progress_bar=noop,
        _collect_advanced_kwargs=lambda: {},
        _conv_history=None,
        _worker=None,
    )


@pytest.mark.skipif(not _GUI_OK, reason=_SKIP_REASON)
def test_wave_optimize_bounds_bracket_metre_x0(monkeypatch):
    """The metre-unit x0 the parameterization starts from must lie inside
    the bounds box for every length variable.  Pre-fix the boxes were
    built in millimetres and excluded x0 (scipy would clip the start);
    post-fix they are metre-scale and bracket it."""
    monkeypatch.setattr(_dock_mod, 'WaveOptimizeWorker', _CaptureWorker)
    _CaptureWorker.last = None

    sm = _build_singlet_pair_model()
    dock = _make_dock_stub(sm)
    _dock_mod.OptimizerDock._start_wave_optimize(dock)

    param = _CaptureWorker.last
    assert param is not None, (
        'the dock did not reach worker construction -- setup raised and '
        'was swallowed by the method-level try/except')

    x0 = param.initial_values()          # metres, read from the template
    bounds = param.bounds
    assert len(x0) == 3 and len(bounds) == 3, (len(x0), len(bounds))

    # Every start value must sit inside its own box (the scipy precond).
    for i, (xi, (lo, hi)) in enumerate(zip(x0, bounds)):
        assert lo <= xi <= hi, (
            f'var {i}: metre x0={xi} outside bounds ({lo}, {hi}) -- '
            f'unit mismatch (mm bounds vs metre x0)')

    # x0 came from to_prescription (metres): radii +-0.05 m, gap 0.02 m.
    assert np.allclose(sorted(x0), [-0.05, 0.02, 0.05], atol=1e-9), x0

    # Independent unit oracle: the OLD bound source (get_variable_values,
    # millimetres) is exactly 1000x the metre x0 -- the two unit systems
    # the bug conflated -- so mm-derived boxes would have been ~1000x too
    # wide and centred 1000x away, excluding every metre x0.
    mm_vals = sm.get_variable_values()   # [50, -50, 20] mm
    assert np.allclose(np.sort(np.abs(mm_vals)),
                       np.sort(np.abs(x0)) * 1e3, rtol=1e-9), (mm_vals, x0)
    # Reconstruct the OLD (mm-sourced) radius box and confirm it would
    # have excluded the metre x0 -- the concrete failure the fix removes.
    old_lo, old_hi = mm_vals[0] * 0.5, mm_vals[0] * 2.0   # 50 -> (25, 100)
    assert not (min(old_lo, old_hi) <= x0[0] <= max(old_lo, old_hi)), (
        x0[0], (old_lo, old_hi))
