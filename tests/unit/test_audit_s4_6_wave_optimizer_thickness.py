"""Audit S4-6 (v5.24.2 exhaustive audit) [P2][seam]: the GUI wave
optimizer could not carry a surface ``thickness`` (or ``semi_diameter``)
variable.  ``_start_wave_optimize`` mapped every non-``distance`` field to
``('surfaces', fs, field)``, but the flattened surface dict emitted by
``ModelState.to_prescription`` has NO ``thickness`` / ``semi_diameter``
key -- so ``DesignParameterization.initial_values()`` (which reads x0 from
the template via ``_read_path``'s direct ``cur[key]`` indexing) raised
``KeyError`` and the flagship "Wave Optimize" button died with "Wave
optimizer setup failed".

Fix under test (lumenairy/ui/optimizer_dock.py, ``_start_wave_optimize``):
route a surface ``thickness`` to its top-level ``thicknesses`` slot
(internal gap for a non-last surface, else the air gap to the following
element), skip fields that have no home in the legacy prescription
(``semi_diameter`` / ``glass``), and de-dup a last-surface thickness that
collides with the next element's ``distance`` on the same gap.

Independent oracle (not a tautology): the test drives the REAL method,
captures the actual ``DesignParameterization`` the dock hands the worker,
and then calls ``param.initial_values()`` -- the exact template read that
KeyError'd pre-fix.  It further pins each mapped ``thicknesses`` slot's x0
against the model's OWN surface thickness/gap (in metres), proving the
slot points at the right gap rather than merely "some" gap.
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
_A_CTR_MM = 5.0     # singlet A centre thickness (internal gap S0->S1)
_GAP_MM = 20.0      # air gap between singlet A and singlet B
_B_R1_MM = 80.0     # singlet B front radius (a plain surface var)


def _build_thickness_var_model():
    """Source + two singlets + detector.  Free vars exercise every S4-6
    branch: an internal-surface ``thickness`` (-> its own ``thicknesses``
    slot), a ``semi_diameter`` (no prescription home -> skipped), a
    LAST-surface ``thickness`` that shares the air gap with the next
    element's ``distance`` (dedup), and a plain surface ``radius``."""
    sm = SystemModel()
    singlet_a = Element(
        1, 'A', 'Singlet', distance_mm=30.0, surfaces=[
            SurfaceRow(radius=50.0, thickness=_A_CTR_MM, glass='N-BK7',
                       semi_diameter=10.0),
            SurfaceRow(radius=-50.0, thickness=0.0, glass='',
                       semi_diameter=10.0),
        ])
    singlet_b = Element(
        2, 'B', 'Singlet', distance_mm=_GAP_MM, surfaces=[
            SurfaceRow(radius=_B_R1_MM, thickness=4.0, glass='N-BK7',
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
        (1, 0, 'thickness'),       # A internal gap  -> ('thicknesses', 0)
        (1, 0, 'semi_diameter'),   # no home         -> skipped
        (1, 1, 'thickness'),       # A->B air gap    -> ('thicknesses', 1)
        (2, 0, 'distance'),        # SAME air gap    -> dedup'd away
        (2, 0, 'radius'),          # plain surface var
    ]
    sm._invalidate()
    return sm, singlet_a, singlet_b


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
    without building the whole Qt widget tree."""
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
def test_wave_optimizer_carries_thickness_and_skips_semidiameter(monkeypatch):
    """Thickness vars reach the optimizer via ``thicknesses`` slots (not a
    KeyError'ing ``('surfaces', fs, 'thickness')`` path); ``semi_diameter``
    is skipped; a last-surface thickness de-dups against the neighbouring
    distance var."""
    monkeypatch.setattr(_dock_mod, 'WaveOptimizeWorker', _CaptureWorker)
    _CaptureWorker.last = None

    sm, singlet_a, singlet_b = _build_thickness_var_model()
    dock = _make_dock_stub(sm)
    _dock_mod.OptimizerDock._start_wave_optimize(dock)

    param = _CaptureWorker.last
    assert param is not None, (
        'the dock did not reach worker construction -- setup raised and '
        'was swallowed by the method-level try/except (the S4-6 KeyError)')

    free_vars = list(param.free_vars)

    # (1) NO surface-thickness path leaks through -- the exact bad path the
    #     fix removes.  Pre-fix free_vars held ('surfaces', 0, 'thickness').
    assert not any(fv[0] == 'surfaces' and fv[-1] == 'thickness'
                   for fv in free_vars), free_vars
    # (2) semi_diameter has no prescription home and must be dropped.
    assert not any(fv[0] == 'surfaces' and fv[-1] == 'semi_diameter'
                   for fv in free_vars), free_vars
    # (3) Both thickness vars land in the top-level ``thicknesses`` list;
    #     the (1,1,'thickness') / (2,0,'distance') pair collapses to ONE
    #     slot -> 5 opt_variables become 3 free vars.
    thk_paths = [fv for fv in free_vars if fv[0] == 'thicknesses']
    assert len(thk_paths) == 2, free_vars
    assert len(free_vars) == 3, free_vars

    # (4) The critical regression: initial_values() reads x0 from the
    #     template at every path.  Pre-fix this KeyError'd on the
    #     ('surfaces', fs, 'thickness') path.  Post-fix it resolves.
    x0 = param.initial_values()
    assert x0.shape == (3,) and np.all(np.isfinite(x0)), x0

    # (5) Independent geometry oracle: each mapped ``thicknesses`` slot's
    #     start value equals the model's OWN thickness/gap in metres --
    #     proving the routing points at the RIGHT gap, not merely a gap.
    slot_val = {fv[1]: float(x0[i])
                for i, fv in enumerate(free_vars) if fv[0] == 'thicknesses'}
    # A internal centre thickness -> slot 0.
    assert np.isclose(slot_val[0], _A_CTR_MM * 1e-3, atol=1e-12), slot_val
    # A->B air gap (== singlet B's distance_mm) -> slot 1.
    assert np.isclose(slot_val[1], _GAP_MM * 1e-3, atol=1e-12), slot_val

    # (6) The surviving surface var is B's front radius, in metres.
    surf_paths = [fv for fv in free_vars if fv[0] == 'surfaces']
    assert len(surf_paths) == 1 and surf_paths[0][-1] == 'radius', free_vars
    r_idx = free_vars.index(surf_paths[0])
    assert np.isclose(float(x0[r_idx]), _B_R1_MM * 1e-3, atol=1e-12), x0

    # Every start value sits inside its own bounds box (scipy precond).
    assert len(param.bounds) == 3
    for i, (xi, (lo, hi)) in enumerate(zip(x0, param.bounds)):
        assert lo <= xi <= hi, (i, xi, (lo, hi))
