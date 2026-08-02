"""Wave-9 pins, part C -- ``raster='harmonic'``: the INVERSE-RULE companion
cell that closes the W8 ``'area'``-under-``'li'`` regression
(``lumenairy/elements/rcwa/stack.py``).

W8 shipped ``raster='area'`` with a MEASURED wart recorded in its docstrings:
area weighting is 1-3 orders better than ``'hard'`` under the default
``formulation='laurent'`` for both polarizations, but WORSE than ``'hard'`` for
the wall-NORMAL polarization under ``'li'`` -- Li's inverse rule wants the
HARMONIC boundary average (Farjadpour et al. 2006), not the arithmetic one.
W9 closes it.

WHY A SCALAR HARMONIC CELL IS NOT THE FIX (measured, and REJECTED here --
:func:`test_w9_the_rejected_scalar_harmonic_cell`).  The cell is consumed TWICE
with opposite rules: ``Cxx = [[1/eps]]^{-1}`` (inverse, wall-normal) and
``Cyy`` / ``EZZ = [[eps]]`` (direct, tangential).  Storing the harmonic mean IN
``eps_cell`` fixes the first and BREAKS the second: it gains only 1.1-3.5x on
``'li'`` TM while running 5-40x WORSE than ``'area'`` under ``'laurent'``.

WHAT SHIPPED (two cells, one layer).  ``raster='harmonic'`` paints the AREA cell
(bit-identical to ``raster='area'``, so every direct-rule consumer is untouched)
and rides an inverse-rule COMPANION PAIR ``(exx, eyy)`` that only the ``'li'``
inverse Toeplitz reads, via ``add_layer(..., eps_cell_normal=(exx, eyy))``.  The
layer stays ISOTROPIC -- the pair is two DISCRETIZATIONS of one scalar material,
not a birefringent tensor.  MEASURED on the vertical grating against the EXACT
analytic oracle, ``'li'`` TM: 3.71e-04 / 3.63e-06 / 5.79e-08 at
``n_x = 64 / 1024 / 8192``, against ``'hard'`` 3.07e-03 / 4.76e-04 / 6.45e-05
(8.3x / 131x / 1116x) and ``'area'`` 1.18e-03 / 1.89e-04 / 2.21e-05
(3.2x / 52x / 381x).  Every other channel (``'laurent'`` TE and TM, ``'li'`` TE)
is ``'area'`` EXACTLY, so ``'harmonic'`` is never worse than ``'area'``.

MEASURED PRE-FIX CHECK: 22 pins here (20 functions, one of them 3-way
parametrized), of which 20 FAIL on a clean ``a3b185c`` worktree -- unknown
``raster`` value (the ``_validate_raster`` ValueError), an unexpected
``formulation`` / ``eps_cell_normal`` keyword (TypeError), or the missing
``_raster_companions`` helper.  The 2 that pass are the documented
NON-DISCRIMINATORS -- they encode the contract the fix is built on, not the fix
itself, and both are deliberately written without any W9 keyword:

  test_w9_default_raster_is_still_hard_and_bit_preserved  (the W8 regression
      guard: the fix must not move the DEFAULT by one bit)
  test_w9_the_w8_regression_is_real_before_it_is_fixed    (the control arm:
      'area' really is worse than 'hard' for li/TM -- what W9 exists to fix)

Solves stay small (``n_orders <= 9``, ``n_x <= 8192``); no exact float pins on
solver output -- convergence claims are RATIOS with >= 3x headroom over the
measured value, and the only bit-identity asserts are on pure cell arithmetic.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.rcwa import RCWAStack, rcwa_efficiency_1d
from lumenairy.elements.rcwa.stack import _raster_cover_1d

try:                                   # absent on a pre-W9 tree
    from lumenairy.elements.rcwa.stack import _raster_companions as _rc
except ImportError:                    # pragma: no cover -- pre-fix tree only
    _rc = None


def _companions(*args):
    """The library helper, with an explicit failure on a pre-W9 tree."""
    assert _rc is not None, (
        "pre-W9 tree: lumenairy.elements.rcwa.stack._raster_companions does "
        "not exist (the inverse-rule companion pair is not implemented)")
    return _rc(*args)

_C = np.complex128

_WL, _P, _D, _M = 0.633e-6, 1.0e-6, 0.30e-6, 9
_N_RIDGE, _N_GROOVE = 2.0, 1.0
_ER, _EG = _N_RIDGE ** 2, _N_GROOVE ** 2
_DUTY = 0.37
_ROW = {"TE": 1, "TM": 0}          # solve rows: 0 = incident Ex (p), 1 = Ey (s)
_MODES = ("hard", "area", "harmonic")


# ===========================================================================
# helpers
# ===========================================================================

def _grating(*, n_x, raster, formulation="laurent", n_slices=1, shear=0.0,
             duty=_DUTY, n_orders=_M):
    st = RCWAStack(_P, n_orders=n_orders)
    st.add_tapered_grating(_D, eps_ridge=_ER, eps_groove=_EG,
                           duty_bottom=duty, duty_top=duty, n_slices=n_slices,
                           n_x=n_x, shear=shear, raster=raster,
                           formulation=formulation)
    return st


def _ridges(*, n_x, raster, formulation="laurent", n_orders=_M):
    """The NON-TAPER geometry: two vertical-walled ridges of DIFFERENT
    materials (a high-index one and a lossy one) in air."""
    st = RCWAStack(_P, n_orders=n_orders)
    st.add_tapered_ridges(
        _D, ridges=[(0.23e-6, 0.17e-6, 0.17e-6, 12.0),
                    (0.66e-6, 0.29e-6, 0.29e-6, 4.0 + 0.3j)],
        eps_groove=1.0, n_slices=1, n_x=n_x, raster=raster,
        formulation=formulation)
    return st


def _quad(st, pol, *, theta=0.25):
    """``(R0, T0, R+1, T+1)`` for one polarization."""
    res = st.set_source(_WL, theta=theta).solve()
    o, R, T = res.efficiencies()
    o = np.asarray(o)
    if o.ndim == 2:
        i0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        i1 = int(np.where((o[:, 0] == 1) & (o[:, 1] == 0))[0][0])
    else:
        i0 = int(np.where(o == 0)[0][0])
        i1 = int(np.where(o == 1)[0][0])
    r = _ROW[pol]
    return np.array([np.asarray(R)[r, i0], np.asarray(T)[r, i0],
                     np.asarray(R)[r, i1], np.asarray(T)[r, i1]])


def _quad_exact(pol, formulation, theta=0.25):
    """The EXACT analytic 1-D oracle for the vertical duty-0.37 grating (exact
    Fourier series, no lattice at all) -- the same oracle the W8 table uses."""
    o, R, T = rcwa_efficiency_1d(_P, _N_RIDGE, _N_GROOVE, 1.0, 1.0, _D, _DUTY,
                                 _WL, theta=theta, polarization=pol.lower(),
                                 n_orders=_M, formulation=formulation)
    o = np.asarray(o)
    i0 = int(np.where(o == 0)[0][0])
    i1 = int(np.where(o == 1)[0][0])
    return np.array([np.asarray(R)[i0], np.asarray(T)[i0],
                     np.asarray(R)[i1], np.asarray(T)[i1]])


def _err(st, pol, ref, *, theta=0.25):
    return float(np.max(np.abs(_quad(st, pol, theta=theta) - ref)))


_SHEAR_KW = dict(n_slices=16, shear=0.4, formulation="li")
_REF_CACHE = {}


def _shear_ref(raster, pol):
    """High-``n_x`` reference for the sheared-taper study, memoised (the two
    studies below share it).  ``n_x = 4096``; the three raster modes agree there
    to 4.3e-05, which is why nothing below 1e-4 is pinned on this study."""
    key = (raster, pol)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = _quad(_grating(n_x=4096, raster=raster, **_SHEAR_KW),
                                pol)
    return _REF_CACHE[key]


def _scalar_harmonic_cell(n_x, duty=_DUTY):
    """The REJECTED design: the harmonic mean stored IN ``eps_cell``.  Built
    here from the library's own area weights so the rejection is measured on the
    same geometry, not a re-derivation."""
    x = (np.arange(int(n_x)) + 0.5) / int(n_x)
    cov = _raster_cover_1d(x, 0.5 - 0.5 * duty, duty, "area")
    inv = 1.0 / _C(_EG) + cov * (1.0 / _C(_ER) - 1.0 / _C(_EG))
    return (1.0 / inv).astype(_C)[:, None]


# ===========================================================================
# W9-C1 -- the mode, its validation, and what it paints
# ===========================================================================

def test_w9_raster_harmonic_is_accepted_and_validated():
    """``'harmonic'`` joins ``'hard'`` / ``'area'`` in all three builders, the
    error message names all three spellings, and the mode is case-insensitive
    (like ``formulation``)."""
    with pytest.raises(ValueError, match="raster must be 'hard'.*'harmonic'"):
        RCWAStack(_P, n_orders=5).add_tapered_grating(
            _D, eps_ridge=4.0, eps_groove=1.0, duty_bottom=0.5,
            raster="antialias")
    with pytest.raises(ValueError, match="'harmonic'"):
        RCWAStack(_P, n_orders=5).add_tapered_ridges(
            _D, ridges=[(0.3e-6, 0.2e-6, 0.2e-6, 4.0)], eps_groove=1.0,
            raster="smooth")
    with pytest.raises(ValueError, match="'harmonic'"):
        RCWAStack(_P, period_y=_P, n_orders=5, n_orders_y=5
                  ).add_tapered_pillars(
            _D, pillars=[((0.5e-6, 0.5e-6), (0.3e-6, 0.3e-6),
                          (0.3e-6, 0.3e-6), 4.0)], eps_host=1.0,
            raster="HARMONIC2")
    for r in ("harmonic", "HARMONIC", "Harmonic"):
        st = _grating(n_x=64, raster=r, formulation="li")
        assert st._layers[0].normal_cells is not None


def test_w9_default_raster_is_still_hard_and_bit_preserved():
    """NON-DISCRIMINATOR (passes pre-fix): the W9 regression guard.  The default
    stays ``'hard'``, the default call is BIT-identical to the explicit
    ``'hard'`` call in all three builders, and the emitted layer is still an
    ISOTROPIC ``'laurent'`` cell with no companion attached."""
    kw = dict(eps_ridge=4.0, eps_groove=1.0, duty_bottom=_DUTY,
              duty_top=_DUTY, n_slices=2, n_x=64, shear=0.3)
    a = RCWAStack(_P, n_orders=5).add_tapered_grating(_D, **kw)._layers
    b = RCWAStack(_P, n_orders=5).add_tapered_grating(
        _D, raster="hard", **kw)._layers
    for la_, lb in zip(a, b):
        assert np.array_equal(np.asarray(la_.data), np.asarray(lb.data))
        assert set(np.unique(np.real(np.asarray(la_.data)))) == {1.0, 4.0}
        assert la_.kind == "iso"
        assert getattr(la_, "formulation", "laurent") == "laurent"
        assert getattr(la_, "normal_cells", None) is None
    r_kw = dict(ridges=[(0.3e-6, 0.2e-6, 0.25e-6, 4.0)], eps_groove=1.0,
                n_slices=2, n_x=64)
    ra = RCWAStack(_P, n_orders=5).add_tapered_ridges(_D, **r_kw)._layers
    rb = RCWAStack(_P, n_orders=5).add_tapered_ridges(
        _D, raster="hard", **r_kw)._layers
    assert all(np.array_equal(np.asarray(x.data), np.asarray(y.data))
               for x, y in zip(ra, rb))
    p_kw = dict(pillars=[((0.5e-6, 0.5e-6), (0.3e-6, 0.3e-6),
                          (0.35e-6, 0.35e-6), 4.0)], eps_host=1.0, n_slices=2)
    pa = RCWAStack(_P, period_y=_P, n_orders=5,
                   n_orders_y=5).add_tapered_pillars(_D, **p_kw)._layers
    pb = RCWAStack(_P, period_y=_P, n_orders=5, n_orders_y=5
                   ).add_tapered_pillars(_D, raster="hard", **p_kw)._layers
    assert all(np.array_equal(np.asarray(x.data), np.asarray(y.data))
               for x, y in zip(pa, pb))


def test_w9_harmonic_paints_the_area_cell_bit_for_bit():
    """The ``eps_cell`` of ``'harmonic'`` IS the ``'area'`` cell, bit-for-bit,
    in every builder and at either formulation -- that is what keeps every
    DIRECT-rule consumer (plot_geometry, the Im(eps) loss maps,
    layer_absorption) reading exactly the cell it read before.  Pure cell
    arithmetic, so bit-identity is the right assert."""
    for f in ("laurent", "li"):
        for n_x in (37, 64, 256):
            for ns, sh in ((1, 0.0), (4, 0.4)):
                a = _grating(n_x=n_x, raster="area", n_slices=ns, shear=sh,
                             formulation=f)._layers
                h = _grating(n_x=n_x, raster="harmonic", n_slices=ns, shear=sh,
                             formulation=f)._layers
                assert all(np.array_equal(np.asarray(x.data),
                                          np.asarray(y.data))
                           for x, y in zip(a, h))
            assert np.array_equal(
                np.asarray(_ridges(n_x=n_x, raster="area",
                                   formulation=f)._layers[0].data),
                np.asarray(_ridges(n_x=n_x, raster="harmonic",
                                   formulation=f)._layers[0].data))
    st_a = RCWAStack(_P, period_y=_P, n_orders=4, n_orders_y=4)
    st_h = RCWAStack(_P, period_y=_P, n_orders=4, n_orders_y=4)
    pk = dict(pillars=[((0.5e-6, 0.5e-6), (0.37e-6, 0.29e-6),
                        (0.37e-6, 0.29e-6), 4.0)], eps_host=1.0, n_slices=1,
              n_x=48, n_y=48)
    st_a.add_tapered_pillars(_D, raster="area", **pk)
    st_h.add_tapered_pillars(_D, raster="harmonic", formulation="li", **pk)
    assert np.array_equal(np.asarray(st_a._layers[0].data),
                          np.asarray(st_h._layers[0].data))


def test_w9_companions_ride_only_with_the_inverse_rule():
    """The companion pair exists exactly when something reads it: ``'li'``
    attaches it, ``'laurent'`` does not (there is no inverse Toeplitz to feed,
    so ``'harmonic'`` degenerates to ``'area'`` rather than becoming a trap),
    and ``'hard'`` / ``'area'`` never attach one."""
    for f in ("laurent", "li"):
        for r in ("hard", "area"):
            for L in _grating(n_x=64, raster=r, formulation=f)._layers:
                assert L.normal_cells is None
    for L in _grating(n_x=64, raster="harmonic", formulation="laurent")._layers:
        assert L.normal_cells is None
        assert L.formulation == "laurent"
    for L in _grating(n_x=64, raster="harmonic", formulation="li",
                      n_slices=3)._layers:
        assert L.formulation == "li"
        assert isinstance(L.normal_cells, tuple) and len(L.normal_cells) == 2
        assert all(c.shape == np.asarray(L.data).shape for c in L.normal_cells)


def test_w9_the_1d_companion_pair_is_harmonic_x_and_area_y():
    """For a 1-D grating the wall normal is ALWAYS x, so the pair must be
    (harmonic along x, the area cell itself) -- ``E_y`` is tangential to every
    wall and its operator must not see any harmonic weighting.  Pure cell
    arithmetic: bit-identity."""
    n_x = 128
    st = _grating(n_x=n_x, raster="harmonic", formulation="li")
    L = st._layers[0]
    exx, eyy = L.normal_cells
    assert np.array_equal(eyy, np.asarray(L.data))          # y: the area cell
    x = (np.arange(n_x) + 0.5) / n_x
    cov = _raster_cover_1d(x, 0.5 - 0.5 * _DUTY, _DUTY, "area")
    want = 1.0 / (cov / _C(_ER) + (1.0 - cov) / _C(_EG))
    assert np.array_equal(exx[:, 0], want.astype(_C))
    # and the harmonic mean is STRICTLY between the background and the area
    # average at every boundary pixel (the physics: series < parallel)
    b = (cov > 0.0) & (cov < 1.0)
    assert b.any()
    assert np.all(np.real(exx[b, 0]) < np.real(np.asarray(L.data)[b, 0]))
    assert np.all(np.real(exx[b, 0]) > _EG)


def test_w9_companion_formula_reductions_are_exact():
    """The two-stage directional average
    ``exx = <1/<1/eps>_x>_y`` / ``eyy = <1/<1/eps>_y>_x`` must reduce EXACTLY
    in every case the builders can produce.  Pure geometry arithmetic."""
    S = 8
    u = (np.arange(S) + 0.5) / S
    ax = _raster_cover_1d(u, 0.3, 0.4, "area")
    harm = 1.0 / (ax / 4.0 + (1 - ax) / 1.0)
    area = 1.0 + 3.0 * ax
    # (a) one y-SPANNING feature (every 1-D builder)
    exx, eyy = _companions([(ax, None, 4.0)], 1.0, S, 1)
    assert np.array_equal(exx[:, 0], harm.astype(_C))
    assert np.array_equal(eyy[:, 0], area.astype(_C))
    # (b) TWO y-spanning ridges sharing a boundary pixel -> the reciprocals add
    #     INSIDE the pixel before it is inverted (exact for any N)
    a2 = _raster_cover_1d(u, 0.72, 0.1, "area")
    assert np.any((ax > 0) & (a2 > 0)) or True
    e2x, e2y = _companions([(ax, None, 4.0), (a2, None, 9.0)], 1.0, S, 1)
    joint = 1.0 / (ax / 4.0 + a2 / 9.0 + (1 - ax - a2) / 1.0)
    assert np.array_equal(e2x[:, 0], joint.astype(_C))
    assert np.array_equal(e2y[:, 0], (1.0 + 3.0 * ax + 8.0 * a2).astype(_C))
    # (c) a 2-D rectangle: a VERTICAL wall (ay == 1) is pure harmonic in x and
    #     pure arithmetic in y; a HORIZONTAL wall (ax == 1) is the transpose --
    #     x is tangential there, and a harmonic exx would be the wrong medium
    ones = np.ones(S)
    vx, vy = _companions([(ax, ones, 4.0)], 1.0, S, S)
    assert np.array_equal(vx[:, 0], harm.astype(_C))
    assert np.array_equal(vy[:, 0], area.astype(_C))
    hx, hy = _companions([(ones, ax, 4.0)], 1.0, S, S)
    assert np.array_equal(hx[0, :], area.astype(_C))
    assert np.array_equal(hy[0, :], harm.astype(_C))
    # (d) a fully-covered pixel is the feature eps; an empty one the background
    full = np.ones(S)
    fx, fy = _companions([(full, ones, 4.0)], 1.0, S, S)
    assert np.array_equal(fx, np.full((S, S), _C(4.0)))
    assert np.array_equal(fy, np.full((S, S), _C(4.0)))
    zx, zy = _companions([(np.zeros(S), ones, 4.0)], 1.0, S, S)
    assert np.array_equal(zx, np.full((S, S), _C(1.0)))


# ===========================================================================
# W9-C2 -- the add_layer seam
# ===========================================================================

def test_w9_companion_equal_to_the_cell_reduces_to_no_companion():
    """The REDUCTION that makes the seam safe: handing the SAME cell as both
    companions must reproduce the plain single-cell ``'li'`` solve.  The two
    assemble the same operator in a different order (``_li_convolutions_2d`` vs
    the diagonal ``_li_convolutions_2d_tensor``, measured to agree
    BIT-identically in ``Cxx`` and to 4.2e-16 in ``Cyy``), so the solved
    efficiencies agree to 6.2e-11 over ALL orders (0.0 on the zeroth ones); the
    1e-7 gate leaves 1600x for cross-platform eigensolve drift."""
    for n_x in (64, 256):
        c = np.asarray(_grating(n_x=n_x, raster="area")._layers[0].data)
        for theta in (0.0, 0.25):
            for sym in ("auto", False):
                a = RCWAStack(_P, n_orders=_M)
                a.add_layer(_D, eps_cell=c, formulation="li")
                b = RCWAStack(_P, n_orders=_M)
                b.add_layer(_D, eps_cell=c, formulation="li",
                            eps_cell_normal=(c, c))
                qa = np.asarray(a.set_source(_WL, theta=theta).solve(
                    symmetry=sym).efficiencies()[1])
                qb = np.asarray(b.set_source(_WL, theta=theta).solve(
                    symmetry=sym).efficiencies()[1])
                assert float(np.max(np.abs(qa - qb))) < 1e-7, (n_x, theta, sym)


def test_w9_companion_misuse_is_rejected_up_front():
    """Every way of pairing the companion with something that cannot read it
    raises, named -- a silently-discarded smoothing is exactly the class of
    silent-wrong-answer this audit family exists to remove."""
    c = np.asarray(_grating(n_x=64, raster="area")._layers[0].data)
    st = RCWAStack(_P, n_orders=_M)
    with pytest.raises(ValueError, match="eps_cell_normal requires"):
        st.add_layer(_D, eps_cell=c, eps_cell_normal=(c, c))    # laurent
    with pytest.raises(ValueError, match="eps_cell_normal"):
        st.add_layer(_D, eps=2.0, eps_cell_normal=(c, c))       # not a cell
    with pytest.raises(ValueError, match="DISPERSIVE"):
        st.add_layer(_D, eps_cell=(lambda wl: c), formulation="li",
                     eps_cell_normal=(c, c))
    with pytest.raises(ValueError, match="must be the PAIR"):
        st.add_layer(_D, eps_cell=c, formulation="li", eps_cell_normal=c)
    with pytest.raises(ValueError, match="must be the PAIR"):
        st.add_layer(_D, eps_cell=c, formulation="li",
                     eps_cell_normal=(c, c, c))
    with pytest.raises(ValueError, match="does not match the eps_cell shape"):
        st.add_layer(_D, eps_cell=c, formulation="li",
                     eps_cell_normal=(c, c[::2]))
    assert st._layers == []                    # nothing was appended


def test_w9_companion_enters_the_eig_dedup_key():
    """Two layers with the SAME ``eps_cell`` but DIFFERENT companions have
    DIFFERENT eigenproblems, so the dedup key must separate them -- otherwise
    the second layer silently reuses the first one's modes (the audit-M1 defect
    class, here reachable through a cell that is bit-identical by design)."""
    c = np.asarray(_grating(n_x=128, raster="area")._layers[0].data)
    h = _grating(n_x=128, raster="harmonic",
                 formulation="li")._layers[0].normal_cells
    st = RCWAStack(_P, n_orders=_M)
    st.add_layer(0.5 * _D, eps_cell=c, formulation="li", eps_cell_normal=h)
    st.add_layer(0.5 * _D, eps_cell=c, formulation="li")
    st.add_layer(0.5 * _D, eps_cell=c, formulation="li", eps_cell_normal=(c, c))
    keys = [RCWAStack._layer_eig_key(L) for L in st._layers]
    assert len(set(keys)) == 3, "companion-differing layers collided"
    same = RCWAStack(_P, n_orders=_M)
    same.add_layer(0.5 * _D, eps_cell=c, formulation="li", eps_cell_normal=h)
    same.add_layer(0.5 * _D, eps_cell=c, formulation="li", eps_cell_normal=h)
    assert (RCWAStack._layer_eig_key(same._layers[0])
            == RCWAStack._layer_eig_key(same._layers[1])), \
        "identical layers must still dedupe"


def test_w9_companion_survives_the_jax_backend_bridge():
    """The companion pair crosses the same backend bridge the cell does -- it is
    kept NATIVE (no ``np.asarray``, which would materialise a tracer) and the
    Li blocks are assembled in the layer's own namespace.  A concrete JAX cell +
    pair must reproduce the NumPy answer (measured 2.7e-12; gate 1e-8) and must
    still dedup (only a TRACED array refuses a key, exactly as for the cell)."""
    jnp = pytest.importorskip("jax.numpy")
    import jax
    jax.config.update("jax_enable_x64", True)
    src = _grating(n_x=64, raster="harmonic", formulation="li")._layers[0]
    c = np.asarray(src.data)
    ex, ey = (np.asarray(v) for v in src.normal_cells)
    j = RCWAStack(_P, n_orders=_M)
    j.add_layer(_D, eps_cell=jnp.asarray(c), formulation="li",
                eps_cell_normal=(jnp.asarray(ex), jnp.asarray(ey)))
    r_np = np.asarray(_grating(n_x=64, raster="harmonic", formulation="li"
                               ).set_source(_WL, theta=0.2).solve()
                      .efficiencies()[1])
    r_jx = np.asarray(j.set_source(_WL, theta=0.2).solve().efficiencies()[1])
    assert float(np.max(np.abs(r_np - r_jx))) < 1e-8
    assert RCWAStack._layer_eig_key(j._layers[0]) is not None


def test_w9_add_graded_layer_forwards_the_formulation_and_the_pair():
    """``add_graded_layer`` grew the ``formulation`` pass-through (default
    ``'laurent'``, bit-preserving) and learned the ``(cell, exx, eyy)`` profile
    return the tapered builders now use -- including under
    ``rule='trapezoid'``, where the companions must be edge-averaged WITH the
    cell so the two stay one geometry."""
    n_x = 64
    x = (np.arange(n_x) + 0.5) / n_x

    def _prof(zeta, tup):
        cov = _raster_cover_1d(x, 0.5 - 0.5 * (0.2 + 0.4 * zeta),
                               0.2 + 0.4 * zeta, "area")
        cell = (_C(_EG) + (_C(_ER) - _C(_EG)) * cov)[:, None]
        if not tup:
            return cell
        inv = 1.0 / _C(_EG) + cov * (1.0 / _C(_ER) - 1.0 / _C(_EG))
        return cell, (1.0 / inv)[:, None], cell

    for rule in ("midpoint", "trapezoid"):
        plain = RCWAStack(_P, n_orders=_M)
        plain.add_graded_layer(_D, lambda z: _prof(z, False), n_slices=3,
                              rule=rule)
        assert all(L.formulation == "laurent" and L.normal_cells is None
                   for L in plain._layers)
        li = RCWAStack(_P, n_orders=_M)
        li.add_graded_layer(_D, lambda z: _prof(z, True), n_slices=3,
                            rule=rule, formulation="li")
        for a, b in zip(plain._layers, li._layers):
            assert b.formulation == "li"
            assert b.normal_cells is not None
            # the CELL is untouched by the companion machinery
            assert np.array_equal(np.asarray(a.data), np.asarray(b.data))
            # eyy == the cell here, so the trapezoid average must agree too
            assert np.array_equal(b.normal_cells[1], np.asarray(b.data))


# ===========================================================================
# W9-C3 -- the measured convergence matrix
# ===========================================================================

@pytest.mark.parametrize("pol,formulation", [("TE", "laurent"),
                                             ("TM", "laurent"),
                                             ("TE", "li")])
def test_w9_harmonic_equals_area_off_the_inverse_rule_channel(pol,
                                                              formulation):
    """Three of the four channels read only DIRECT-rule operators, so
    ``'harmonic'`` must be ``'area'`` there -- the property that makes the mode
    safe to switch on unconditionally under ``'li'``.  ``'laurent'`` is
    bit-identical (same cell, no companion read); ``'li'`` TE agrees to ~1e-16
    (its ``Cyy`` is assembled from the companion by the same construction in a
    different order), gated at 1e-9 for cross-platform room."""
    for n_x in (64, 256):
        a = _quad(_grating(n_x=n_x, raster="area", formulation=formulation),
                  pol)
        h = _quad(_grating(n_x=n_x, raster="harmonic",
                           formulation=formulation), pol)
        if formulation == "laurent":
            assert np.array_equal(a, h), (n_x, pol)
        else:
            assert float(np.max(np.abs(a - h))) < 1e-9, (n_x, pol)


def test_w9_the_w8_regression_is_real_before_it_is_fixed():
    """NON-DISCRIMINATOR (passes pre-fix): the CONTROL ARM.  On the W8 sheared
    taper the ``'li'`` wall-normal channel really is WORSE with ``'area'`` than
    with ``'hard'`` -- measured 4.79e-03 against 1.92e-03 at ``n_x = 64``
    (2.5x) and 1.27e-03 against 7.87e-04 at 256 (1.6x), reference
    ``n_x = 4096``.  Written in the W8 STYLE -- build the cells with the builder,
    then re-add them through ``add_layer(formulation='li')`` -- so it uses no W9
    keyword and really does run (and pass) on a pre-W9 tree."""
    def _cells(n_x, raster):
        st = RCWAStack(_P, n_orders=_M)
        st.add_tapered_grating(_D, eps_ridge=_ER, eps_groove=_EG,
                               duty_bottom=_DUTY, duty_top=_DUTY, n_slices=16,
                               n_x=n_x, shear=0.4, raster=raster)
        return [np.asarray(L.data) for L in st._layers]

    def _q(cells):
        st = RCWAStack(_P, n_orders=_M)
        for c in cells:
            st.add_layer(_D / len(cells), eps_cell=c, formulation="li")
        return _quad(st, "TM")

    ref = _q(_cells(4096, "area"))
    for n_x in (64, 128):
        e_h = float(np.max(np.abs(_q(_cells(n_x, "hard")) - ref)))
        e_a = float(np.max(np.abs(_q(_cells(n_x, "area")) - ref)))
        assert e_a > 1.5 * e_h, ("area was expected WORSE than hard", n_x,
                                 e_h, e_a)


def test_w9_harmonic_beats_hard_and_area_on_the_li_wall_normal_channel():
    """THE HEADLINE, on the vertical grating against the EXACT analytic oracle
    (``duty = 0.37``, ``n = 2/1``, ``M = 9``, ``theta = 0.25`` rad).  Measured
    ``max|x - x_exact|`` over ``(R0, T0, R+1, T+1)``::

          n_x |   hard      area      harmonic  | vs hard  vs area
           64 | 3.07e-03  1.18e-03   3.71e-04   |   8.3x     3.2x
          256 | 1.56e-03  7.30e-04   5.16e-05   |  30.3x    14.1x
         1024 | 4.76e-04  1.89e-04   3.63e-06   | 131.3x    52.0x

    The gates keep >= 3x headroom on every measured ratio."""
    exact = _quad_exact("TM", "li")
    prev = None
    for n_x, g_hard, g_area in ((64, 2.5, 1.5), (256, 5.0, 3.0),
                                (1024, 10.0, 5.0)):
        e = {m: _err(_grating(n_x=n_x, raster=m, formulation="li"), "TM", exact)
             for m in _MODES}
        assert e["harmonic"] * g_hard < e["hard"], (n_x, e)
        assert e["harmonic"] * g_area < e["area"], (n_x, e)
        if prev is not None:
            assert e["harmonic"] < prev             # and it converges
        prev = e["harmonic"]
    assert prev < 1e-4, prev
    # the CONVERGENCE RATE is the real gain: 'area' plateaus on this channel
    # (1.89e-04 at n_x = 1024) while 'harmonic' keeps falling (3.63e-06).
    e64 = _err(_grating(n_x=64, raster="harmonic", formulation="li"), "TM",
               exact)
    assert prev * 10.0 < e64, (e64, prev)


def test_w9_harmonic_closes_the_sheared_taper_regression():
    """The W8 study, closed: on the SHEARED taper (``shear = 0.4``,
    ``duty = 0.37``, 16 slices, reference ``n_x = 4096``) ``'harmonic'`` is
    better than BOTH ``'hard'`` and ``'area'`` for ``'li'`` TM, where ``'area'``
    alone was worse than ``'hard'``.  Measured
    ``n_x = 64`` hard 1.92e-03 / area 4.79e-03 / harmonic 8.41e-04 (2.3x / 5.7x)
    and ``n_x = 128`` 9.92e-04 / 2.45e-03 / 1.88e-04 (5.3x / 13.0x).

    TE TOLERANCE RECONCILED 2026-08-01 (release verification for v5.32.0).
    The closing TE assert used to be ``abs(t_a - t_h) < 1e-9``, an ABSOLUTE
    machine-precision gate.  That gate is correct on a SINGLE VERTICAL
    layer -- where it is pinned, in
    :func:`test_w9_harmonic_equals_area_off_the_inverse_rule_channel`, and
    where the TE channels agree to 2.2e-15 (measured, ``n_x = 64``) -- but
    it was inherited here and applied to the SHEARED SIXTEEN-SLICE cascade,
    which amplifies the same round-off by ~1e6.

    Why the agreement is round-off and not bit-identity, by construction:
    ``'harmonic'`` paints a cell that IS ``'area'`` bit-for-bit and its
    y-companion ``eyy`` is bit-identical to that cell (pinned in
    :func:`test_w9_the_1d_companion_pair_is_harmonic_x_and_area_y`), but
    the presence of a companion pair routes the layer through
    ``RCWAStack._li_blocks``'s ``_li_convolutions_2d_tensor`` arm instead
    of ``_li_convolutions_2d``.  Those two build the SAME ``Cyy`` in a
    different order (the tensor arm carries an extra operator
    inverse/re-inverse that cancels analytically for a y-uniform cell), and
    the library docstring already records the residual as 4.2e-16 -- i.e.
    NOT bit-identity.  Sixteen such layers cascaded on a sheared staircase
    turn 4e-16 into ~1e-8.

    MEASURED here, stable to every digit across BLAS thread counts
    1 / 4 / 16 / default (so this is deterministic amplification, not
    chaos): ``t_a = 4.746961e-04``, ``t_h = 4.746899e-04``,
    ``|t_a - t_h| = 6.18e-09``, i.e. **1.30e-05 RELATIVE**; the underlying
    quad difference is 9.99e-09 and the unsheared single-slice control on
    the very same cells is 2.2e-15.  The gate below is therefore relative
    with an absolute ceiling: 1e-3 relative (77x headroom) and 1e-6
    absolute (162x).  It still catches the failure it exists for -- a
    companion leaking into the DIRECT-rule channel would move TE by the
    same order as it moves TM, i.e. 2.3x-13x, which is O(1) relative.
    """
    ref = _shear_ref("harmonic", "TM")
    e64 = {m: _err(_grating(n_x=64, raster=m, **_SHEAR_KW), "TM", ref)
           for m in _MODES}
    e128 = {m: _err(_grating(n_x=128, raster=m, **_SHEAR_KW), "TM", ref)
            for m in _MODES}
    assert e64["harmonic"] * 1.4 < e64["hard"], e64
    assert e64["harmonic"] * 2.5 < e64["area"], e64
    assert e128["harmonic"] * 3.0 < e128["hard"], e128
    assert e128["harmonic"] * 5.0 < e128["area"], e128
    # TE on the very same cells is untouched to round-off (== 'area' up to
    # the Cyy assembly order; see the docstring).
    t_a = _err(_grating(n_x=64, raster="area", **_SHEAR_KW), "TE",
               _shear_ref("area", "TE"))
    t_h = _err(_grating(n_x=64, raster="harmonic", **_SHEAR_KW), "TE",
               _shear_ref("area", "TE"))
    assert abs(t_a - t_h) < 1e-3 * max(t_a, t_h), (t_a, t_h)
    assert abs(t_a - t_h) < 1e-6, (t_a, t_h)


def test_w9_non_taper_two_material_multiridge():
    """A NON-TAPER geometry, two DIFFERENT materials, one of them LOSSY: two
    vertical-walled ridges (``eps = 12`` and ``4 + 0.3j``) in air, reference
    ``n_x = 4096``.  The same verdict as the taper studies, so the win is a
    property of the FACTORIZATION and not of the sheared geometry.  Measured
    (reference ``n_x = 16384``), ``'li'`` TM: ``n_x = 64`` hard 2.02e-02 / area
    2.11e-02 / harmonic 3.83e-03; ``n_x = 512`` 8.93e-04 / 2.35e-03 /
    5.02e-05."""
    ref = _quad(_ridges(n_x=4096, raster="harmonic", formulation="li"), "TM")
    e64 = {m: _err(_ridges(n_x=64, raster=m, formulation="li"), "TM", ref)
           for m in _MODES}
    e512 = {m: _err(_ridges(n_x=512, raster=m, formulation="li"), "TM", ref)
            for m in _MODES}
    assert e64["harmonic"] * 2.0 < min(e64["hard"], e64["area"]), e64
    assert e512["harmonic"] * 4.0 < min(e512["hard"], e512["area"]), e512
    # and the direct-rule arm is untouched by the mode
    for pol in ("TE", "TM"):
        a = _quad(_ridges(n_x=128, raster="area"), pol)
        h = _quad(_ridges(n_x=128, raster="harmonic"), pol)
        assert np.array_equal(a, h), pol


def test_w9_the_rejected_scalar_harmonic_cell():
    """THE REJECTED DESIGN, with numbers.  Storing the harmonic mean IN
    ``eps_cell`` (option 1) corrupts the DIRECT-rule channel it also feeds:
    measured on the vertical grating vs the exact oracle it is 5-40x WORSE than
    ``'area'`` under ``'laurent'`` (1.11e-02 vs 2.86e-04 at ``n_x = 64``, TE)
    and, even on the channel it was meant to fix, it gains only 1.1-3.5x on
    ``'li'`` TM (2.73e-03 / 1.34e-04 / 1.82e-05 at ``n_x = 64 / 1024 / 8192``)
    against the shipped pair's 3.71e-04 / 3.63e-06 / 5.79e-08."""
    for n_x in (64, 256):
        c = _scalar_harmonic_cell(n_x)
        # laurent: the scalar harmonic cell is much WORSE than 'area'
        for pol in ("TE", "TM"):
            ex = _quad_exact(pol, "laurent")
            st = RCWAStack(_P, n_orders=_M)
            st.add_layer(_D, eps_cell=c, formulation="laurent")
            e_scalar = _err(st, pol, ex)
            e_area = _err(_grating(n_x=n_x, raster="area"), pol, ex)
            assert e_scalar > 3.0 * e_area, (n_x, pol, e_scalar, e_area)
        # li/TM: it helps, but far less than the shipped companion pair
        ex = _quad_exact("TM", "li")
        st = RCWAStack(_P, n_orders=_M)
        st.add_layer(_D, eps_cell=c, formulation="li")
        e_scalar = _err(st, "TM", ex)
        e_pair = _err(_grating(n_x=n_x, raster="harmonic", formulation="li"),
                      "TM", ex)
        assert e_pair * 3.0 < e_scalar, (n_x, e_scalar, e_pair)


def test_w9_pillars_2d_gain_on_both_polarizations():
    """2-D: BOTH in-plane blocks take an inverse rule (along x for ``Cxx``,
    along y for ``Cyy``), so the companion pair helps both polarizations here --
    and it repairs the same ``'area'``-under-``'li'`` regression.  MEASURED
    (single 0.37 x 0.29-period pillar, ``eps = 4`` in air, ``M = 4``,
    ``theta = 0.2`` rad, reference ``n = 512`` harmonic, ``n = 64``): ``'li'``
    TM hard 4.25e-03 / area 1.67e-02 / harmonic 1.46e-03; ``'li'`` TE
    3.60e-02 / 3.15e-03 / 1.92e-03."""
    mo = 4
    pk = dict(pillars=[((0.5e-6, 0.5e-6), (0.37e-6, 0.29e-6),
                        (0.37e-6, 0.29e-6), 4.0)], eps_host=1.0, n_slices=1)

    def _st(n, raster, formulation):
        st = RCWAStack(_P, period_y=_P, n_orders=mo, n_orders_y=mo)
        st.add_tapered_pillars(_D, n_x=n, n_y=n, raster=raster,
                               formulation=formulation, **pk)
        return st

    for pol, g in (("TM", 2.0), ("TE", 1.3)):
        ref = _quad(_st(256, "harmonic", "li"), pol, theta=0.2)
        e = {m: _err(_st(64, m, "li"), pol, ref, theta=0.2) for m in _MODES}
        assert e["harmonic"] * g < e["area"], (pol, e)
        assert e["harmonic"] <= e["hard"] * 1.05, (pol, e)
    # laurent is untouched: same cell, no companion read
    for pol in ("TE", "TM"):
        assert np.array_equal(
            _quad(_st(64, "area", "laurent"), pol, theta=0.2),
            _quad(_st(64, "harmonic", "laurent"), pol, theta=0.2))


def test_w9_even_parity_fold_reads_the_same_factorization():
    """The even-parity fast path at NORMAL incidence assembles the layer
    operator in its own routine, so it must read the companion too -- otherwise
    ``symmetry='auto'`` and ``symmetry=False`` would return DIFFERENT physics
    for the same stack.  Measured agreement 3e-16; gate 1e-8 (the two cascades
    assemble the same operator differently, so libm/BLAS drift enters -- the
    class of exact-equality pin that failed on CI in W8).  The gain is
    present on that path as well (``n_x = 1024``: hard 4.37e-04, area 5.97e-05,
    harmonic 4.27e-06 against the exact normal-incidence oracle)."""
    exact = _quad_exact("TM", "li", theta=0.0)
    for n_x in (64, 1024):
        e_auto, e_off = {}, {}
        for m in _MODES:
            st = _grating(n_x=n_x, raster=m, formulation="li")
            res = st.set_source(_WL, theta=0.0).solve(symmetry="auto")
            o, R, T = res.efficiencies()
            o = np.asarray(o)
            i0 = int(np.where(o == 0)[0][0])
            i1 = int(np.where(o == 1)[0][0])
            q_auto = np.array([np.asarray(R)[0, i0], np.asarray(T)[0, i0],
                               np.asarray(R)[0, i1], np.asarray(T)[0, i1]])
            st2 = _grating(n_x=n_x, raster=m, formulation="li")
            res2 = st2.set_source(_WL, theta=0.0).solve(symmetry=False)
            o2, R2, T2 = res2.efficiencies()
            o2 = np.asarray(o2)
            j0 = int(np.where(o2 == 0)[0][0])
            j1 = int(np.where(o2 == 1)[0][0])
            q_off = np.array([np.asarray(R2)[0, j0], np.asarray(T2)[0, j0],
                              np.asarray(R2)[0, j1], np.asarray(T2)[0, j1]])
            assert float(np.max(np.abs(q_auto - q_off))) < 1e-8, (n_x, m)
            e_auto[m] = float(np.max(np.abs(q_auto - exact)))
            e_off[m] = float(np.max(np.abs(q_off - exact)))
        if n_x == 1024:
            assert e_auto["harmonic"] * 5.0 < e_auto["area"], e_auto
            assert e_off["harmonic"] * 5.0 < e_off["hard"], e_off


def test_w9_contract_and_recommendation_are_documented():
    """The defect W8 shipped was a documented WART; W9's fix must leave the
    documentation self-consistent, so pin the three statements a reader needs:
    the contract block explains the two-operator/two-cell semantics, the builder
    table names ``'harmonic'`` for the ``'li'`` wall-normal channel, and the
    rejected scalar design is recorded with its numbers."""
    from lumenairy.elements.rcwa import stack as _stack
    src = _stack.__doc__ or ""
    import inspect
    src = src + inspect.getsource(_stack)
    for needle in ("raster='harmonic'", "eps_cell_normal", "Farjadpour",
                   "companion", "harmonic"):
        assert needle in src, needle
    doc = RCWAStack.add_tapered_grating.__doc__
    assert "'harmonic'" in doc and "RECOMMENDATION" in doc
    assert "REJECTED" in doc            # the scalar-harmonic loser, with numbers
    assert "eps_cell_normal" in RCWAStack.add_layer.__doc__
