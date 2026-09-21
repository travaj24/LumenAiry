"""PMM2DStackPure geometry viewers (docs/audits/AUDIT_PMM2D_PURE_VIEWER_2026_09_21.md).

The pure staggered 2-D family was the only stack family in the library with no geometry viewer:
PMMStack, RCWAStack, SegmentStackGeometry and PMM2DStackHybrid all have one, and the contract they
keep is that the picture is READ OUT OF THE RECORD THE SOLVE CONSUMES, so a figure cannot drift
from the physics it claims to show.  These gates pin that contract, the material styling, and the
refusals -- and each one is written so it FAILS if the drawing is re-derived instead of read.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from lumenairy.elements.pmm.stack2d_pure import (
    PMM2DStackPure,
    _stag_edge_map,
    _stag_eps_key,
    _stag_eps_label,
    _stag_material_style,
)

@pytest.fixture(autouse=True)
def _close_figures():
    """Every gate here draws; leaking figures across a suite is a slow memory bug."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


_PX, _PY = 600e-9, 480e-9
_EPS_CU = complex(-83.131, 2.702)          # copper at 1310 nm
_EPS_LC_O = 1.56 ** 2


def _tensor(exx, eyy, exy=0.0):
    t = np.zeros((3, 3), dtype=complex)
    t[0, 0] = exx
    t[1, 1] = eyy
    t[2, 2] = exx
    t[0, 1] = t[1, 0] = exy
    return t


def _stack(walls_x=(180e-9, 420e-9), walls_y=(160e-9, 320e-9), slant=None):
    s = PMM2DStackPure(_PX, _PY, n_superstrate=1.56, n_substrate=1.0,
                       n_modes=5, n_orders=3, layer_grids="per-layer")
    cell = np.zeros((3, 3, 3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            cell[i, j] = (_EPS_CU if (i + j) % 2 == 0 else _EPS_LC_O) * np.eye(3)
    cell[1, 1] = _tensor(_EPS_LC_O, 1.90 ** 2)          # the anisotropic cell
    kw = {} if slant is None else dict(slant=slant)
    s.add_layer(200e-9, eps_cell=cell, x_walls=np.array(walls_x),
                y_walls=np.array(walls_y), **kw)
    s.add_layer(40e-9, eps=3.17)
    return s


# ======================================================================== #
# 1 -- the picture is the record, not a re-derivation
# ======================================================================== #

def test_plan_rectangles_sit_on_the_stored_walls():
    """Every drawn edge must be a wall the solver uses -- to the last digit."""
    wx, wy = (137e-9, 401e-9), (211e-9, 333e-9)
    s = _stack(wx, wy)
    axes = s.plot_geometry()
    xs = sorted({round(p.get_x(), 15) for p in axes[0].patches})
    ys = sorted({round(p.get_y(), 15) for p in axes[0].patches})
    assert xs == [0.0, round(wx[0], 15), round(wx[1], 15)]
    assert ys == [0.0, round(wy[0], 15), round(wy[1], 15)]
    widths = sorted({round(p.get_width(), 15) for p in axes[0].patches})
    assert widths == sorted({round(v, 15) for v in
                             (wx[0], wx[1] - wx[0], _PX - wx[1])})


def test_a_uniform_layer_draws_one_rectangle_per_panel():
    s = _stack()
    axes = s.plot_geometry()
    assert len(axes) == len(s._layers) == 2
    assert len(axes[0].patches) == 9          # 3 x 3 segments
    assert len(axes[1].patches) == 1          # the uniform layer


def test_section_thickness_equals_the_stack():
    """The section must account for every layer, with nothing dropped or doubled."""
    s = _stack()
    ax = s.plot_section()
    zs = [min(v[1] for v in p.get_xy()) for p in ax.patches
          if hasattr(p, "get_xy") and np.asarray(p.get_xy()).ndim == 2]
    total = sum(L["thickness"] for L in s._layers)
    assert min(zs) == pytest.approx(-total, rel=0, abs=1e-18)


def test_section_cut_selects_the_row_the_solver_would_use():
    """Two cuts through different y rows must show different materials."""
    s = _stack()
    lo = s.plot_section(y=80e-9)
    hi = s.plot_section(y=240e-9)
    fc_lo = sorted({tuple(np.round(p.get_facecolor(), 6)) for p in lo.patches})
    fc_hi = sorted({tuple(np.round(p.get_facecolor(), 6)) for p in hi.patches})
    assert fc_lo != fc_hi


def test_editing_the_record_changes_the_picture():
    """The strongest form of the contract: mutate what the solve would read, and the drawing
    must follow.  A viewer that cached or re-derived its own geometry fails here."""
    s = _stack()
    before = sorted({tuple(np.round(p.get_facecolor(), 6))
                     for p in s.plot_geometry()[0].patches})
    s._layers[0]["eps_cell"][0, 0] = 2.10 * np.eye(3)
    after = sorted({tuple(np.round(p.get_facecolor(), 6))
                    for p in s.plot_geometry()[0].patches})
    assert before != after


# ======================================================================== #
# 2 -- material styling is keyed on the physics
# ======================================================================== #

def test_a_metal_is_drawn_warm_and_a_dielectric_cool():
    """The defect this replaces: a categorical palette that coloured copper green.  A metal must
    land in the copper-to-bronze family and a glass-like dielectric in the cool one."""
    cu, cu_hatch = _stag_material_style(_EPS_CU)
    si3n4, _ = _stag_material_style(1.996 ** 2)
    assert cu[0] - cu[2] > 0.2, "a metal must be warm (red above blue)"
    assert si3n4[2] - si3n4[0] > 0.05, "a dielectric of index 2 must be cool"
    assert cu_hatch == ""


def test_a_stronger_metal_is_drawn_darker():
    weak, _ = _stag_material_style(complex(-5.0, 1.0))
    strong, _ = _stag_material_style(complex(-140.0, 3.0))
    assert sum(strong) < sum(weak)


def test_an_anisotropic_cell_is_hatched_and_a_scalar_is_not():
    """A tensor must never be mistakable for the scalar sharing its eps_xx."""
    iso, h_iso = _stag_material_style(_EPS_LC_O)
    ten, h_ten = _stag_material_style(_tensor(_EPS_LC_O, 1.90 ** 2))
    assert h_iso == "" and h_ten != ""
    assert _stag_eps_key(_EPS_LC_O) != _stag_eps_key(_tensor(_EPS_LC_O, 1.90 ** 2))


def test_two_director_angles_do_not_merge_in_the_legend():
    a = _tensor(2.43, 3.61, 0.0)
    b = _tensor(3.02, 3.02, 0.588)          # the same crystal at 45 degrees
    assert _stag_eps_key(a) != _stag_eps_key(b)
    assert "tensor" in _stag_eps_label(_stag_eps_key(a))


def test_near_identical_dielectrics_are_separated_by_edge_colour():
    """Fill is keyed on the physics, so two materials 2 % apart in index MUST look 2 % apart.
    Identity therefore rides on the edge: every distinct material in a figure takes the next
    edge colour, so alumina and carbonitride are still tellable apart."""
    s = PMM2DStackPure(_PX, _PX, n_modes=5, n_orders=3, layer_grids="per-layer")
    cell = np.zeros((2, 2), dtype=complex)
    cell[0, 0] = cell[1, 1] = 1.746 ** 2          # alumina
    cell[0, 1] = cell[1, 0] = 1.781 ** 2          # carbonitride, 2 % away
    s.add_layer(100e-9, eps_cell=cell)
    ax = s.plot_geometry()[0]
    faces = {tuple(np.round(p.get_facecolor()[:3], 3)) for p in ax.patches}
    edges = {tuple(np.round(p.get_edgecolor()[:3], 3)) for p in ax.patches}
    assert len(faces) == 2, "two materials must still have two fills"
    assert max(abs(a - b) for a, b in zip(*sorted(faces))) < 0.06,         "and those fills must be close, because the materials are"
    assert len(edges) == 2, "the edge must separate what the fill cannot"


def test_the_dielectric_ramp_separates_the_crowded_band():
    """A single two-colour ramp flattened everything between index 1.7 and 2.0, which is where
    real dielectrics live.  The multi-stop ramp must keep them apart."""
    al2o3, _ = _stag_material_style(1.746 ** 2)
    si3n4, _ = _stag_material_style(1.996 ** 2)
    sep = max(abs(a - b) for a, b in zip(al2o3, si3n4))
    assert sep > 0.15, f"alumina and nitride must be distinguishable, got {sep:.3f}"


def test_one_material_keeps_one_edge_across_every_panel():
    """The defect this replaces: edges assigned in order of first appearance gave the SAME
    material different edges in two panels of one figure.  Assignment is stack-wide, so a
    material's edge is fixed for the whole drawing."""
    s = _stack()
    axes = s.plot_geometry()
    sec = s.plot_section(y=200e-9)

    def by_face(patches):
        out = {}
        for p in patches:
            out.setdefault(tuple(np.round(p.get_facecolor()[:3], 4)),
                           set()).add(tuple(np.round(p.get_edgecolor()[:3], 4)))
        return out

    seen = {}
    for src in list(axes) + [sec]:
        for face, edge_set in by_face(src.patches).items():
            assert len(edge_set) == 1, "one material, one edge within a panel"
            seen.setdefault(face, set()).update(edge_set)
    for face, edge_set in seen.items():
        assert len(edge_set) == 1, f"material {face} changed edge between panels"


def test_every_material_in_a_stack_gets_its_own_edge():
    s = _stack()
    edges = _stag_edge_map(s._layers)
    assert len(set(edges.values())) == len(edges) >= 3


def test_named_material_takes_the_given_colour():
    s = _stack()
    axes = s.plot_geometry(material_names={_EPS_CU: "Cu"},
                           material_colors={"Cu": (0.0, 0.0, 0.0, 1.0)})
    assert any(tuple(np.round(p.get_facecolor(), 6)) == (0.0, 0.0, 0.0, 1.0)
               for p in axes[0].patches)


# ======================================================================== #
# 3 -- slant is drawn, and drawn with the documented sign
# ======================================================================== #

def test_slant_shears_the_section_by_thickness_times_slant():
    t, sl = 200e-9, 0.08
    s = _stack(slant=(sl, 0.0))
    ax = s.plot_section()
    poly = [np.asarray(p.get_xy()) for p in ax.patches
            if hasattr(p, "get_xy") and np.asarray(p.get_xy()).ndim == 2
            and np.asarray(p.get_xy()).shape[0] >= 4]
    sheared = [p for p in poly if abs(p[0, 0] - p[3, 0]) > 1e-15]
    assert sheared, "a slanted layer must not be drawn as a rectangle"
    offs = {round(float(p[0, 0] - p[3, 0]), 15) for p in sheared}
    assert offs == {round(t * sl, 15)}


def test_reversing_the_slant_reverses_the_drawn_shear():
    a = _stack(slant=(0.08, 0.0)).plot_section()
    b = _stack(slant=(-0.08, 0.0)).plot_section()

    def first_shear(ax):
        for p in ax.patches:
            if hasattr(p, "get_xy"):
                v = np.asarray(p.get_xy())
                if v.ndim == 2 and v.shape[0] >= 4 and abs(v[0, 0] - v[3, 0]) > 1e-15:
                    return float(v[0, 0] - v[3, 0])
        return 0.0

    assert first_shear(a) == pytest.approx(-first_shear(b), rel=1e-12)


def test_a_vertical_layer_is_drawn_vertical():
    ax = _stack().plot_section()
    for p in ax.patches:
        if hasattr(p, "get_xy"):
            v = np.asarray(p.get_xy())
            if v.ndim == 2 and v.shape[0] >= 4:
                assert abs(v[0, 0] - v[3, 0]) < 1e-18


# ======================================================================== #
# 4 -- refusals, and independence from the source
# ======================================================================== #

def test_an_empty_stack_is_refused_by_name():
    s = PMM2DStackPure(_PX, _PY, n_modes=5, n_orders=3)
    with pytest.raises(ValueError, match="add layers first"):
        s.plot_geometry()
    with pytest.raises(ValueError, match="add layers first"):
        s.plot_section()


def test_a_bad_axis_is_refused():
    with pytest.raises(ValueError, match="along must be"):
        _stack().plot_section(along="z")


def test_plotting_needs_no_source_and_leaves_the_solve_untouched():
    """Geometry is complete before set_source, and drawing must not disturb the stack."""
    s = _stack()
    s.plot_geometry()
    s.plot_section()
    s.set_source(1310e-9, theta=0.1, phi=0.3)
    J1 = np.asarray(s.solve(jones=True)[3])
    s.plot_section(y=10e-9)
    J2 = np.asarray(s.solve(jones=True)[3])
    assert np.array_equal(J1, J2)


def test_shared_grid_stacks_draw_too():
    """The viewers must work on the union-grid path, which stores no wall array."""
    s = PMM2DStackPure(_PX, _PX, n_modes=5, n_orders=3)
    cell = np.full((2, 2), _EPS_LC_O, dtype=complex)
    cell[0, 0] = _EPS_CU
    s.add_layer(150e-9, eps_cell=cell)
    axes = s.plot_geometry()
    assert len(axes[0].patches) == 4
    xs = sorted({round(p.get_x(), 15) for p in axes[0].patches})
    assert xs == [0.0, round(_PX / 2, 15)]
