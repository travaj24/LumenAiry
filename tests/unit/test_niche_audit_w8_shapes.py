"""Audit W8 (2026-07-27) -- the ANALYTIC form-factor path of the 2-D RCWA
crossed-grating solver, cross-validated against the rasterised (pixel-cell)
path and against oracle-free closed-form identities.

The W7 rcwa audit closed every recorded non-coverage item except one:
``rcwa_efficiency_2d_shapes``' analytic form-factor path "was read but not
numerically cross-validated against the rasterised path", while the campaign's
standing recommendation ("for accuracy-critical geometry use the shapes path --
exact form factors, no rasterization") rested on exactly that validation.  This
module supplies it, for every shape kind ``_shape_form_factor`` supports
(``rectangle`` / ``disk`` / ``ellipse``), lossless and lossy, normal and
conical, at two fill fractions each.

VERDICT -- the form factors themselves are CLEAN.  The DC coefficient is the
closed-form area fraction to 0.0, a cell-filling rectangle reproduces the
uniform cell exactly, two abutting rectangles reproduce their merged rectangle
to 1.1e-16, periodic wrap is exact (not asymptotic), and the rasterised path
converges TO the analytic answer at the sampling order -- ``O(1/S)`` for a
point-sampled cell, ``O(1/S^2)`` for an area-averaged one -- with no systematic
residual (down to 8.4e-6 in R/T and 5.4e-5 in ``[[eps]]``).

Three defects were in the PLUMBING around them, all silent and all
energy-clean, now guarded in :func:`_validate_shapes`:

* **W8-A** the docstring promised "shapes are painted in order over the
  background" while the analytic factorization ADDS the form factors, which is
  the same thing only on a DISJOINT list.  Overlapping shapes were accepted
  whenever their total area still fitted in one cell (the cumulative-area guard
  is blind to the arrangement) and silently double-counted the shared area:
  measured R/T off by 6.1e-2 (two 5/6-overlapping rectangles), 1.3e-1 (two
  partially overlapping disks) and 1.1e-1 (two IDENTICAL disks -- the case
  AUDIT_V5_5_2 2026-05-31 recorded and the v5.5.3 cumulative-fraction fix did
  not reach), with energy closures of -6.7e-16 / -3.8e-15 / -1.9e-14.
* **W8-B** ``n_orders_y = 0`` on a y-VARYING shape list silently solved the
  y-AVERAGED structure -- R00 = 0.054846 for a disk against the y-resolved
  0.006897 (8x), closure 4.4e-16 -- the exact trap the pixel path has rejected
  since audit M8 via ``_validate_cell_sampling(strict_y=True)``; the analytic
  path had no counterpart (nor did ``RCWAStack.add_layer(shapes=...)``).  Its
  1-D-stack half (**W8-B'**) had no counterpart either: where a y-varying
  ``eps_cell`` on a 1-D stack has emitted ``RCWAYAverageWarning`` since commit
  ``809314c``, a y-varying SHAPE list was silent -- measured through the stack
  API, R00 = 0.054846364 against the 2-D stack's 0.006896833 (7.95x, absolute
  0.047950), both closures ~1e-15.  Both halves now read ONE predicate
  (``_shapes_y_varying``), so raise and warn cannot diverge -- the same
  no-divergence contract ``809314c`` pinned for the pixel pair.
* **W8-C** a malformed shape dict surfaced as a bare ``KeyError('radius')`` /
  ``TypeError`` from inside the form factors instead of a named ``ValueError``.

RASTER ORACLE.  ``eps_cell[j, i]`` is a POINT SAMPLE on the NODE lattice
``arange(S)/S`` (the pixel contract that :func:`_eps_convolution_2d` defines --
audit W8's raster half), so :func:`_raster` samples there and wraps in both
axes; ``sub > 1`` averages a ``sub x sub`` sub-grid per pixel, i.e. the
area-averaged rasterization a careful caller writes.  Both are legal cells; they
differ only in convergence ORDER, and both converge to the analytic answer,
which is the statement under test.
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements.rcwa import (
    _analytic_convolutions_2d,
    _eps_convolution_2d,
    _harmonic_orders_2d,
    _shape_form_factor,
    _shapes_overlap,
    _shapes_y_varying,
    _validate_shapes,
    rcwa_efficiency_2d,
    rcwa_efficiency_2d_shapes,
)
from lumenairy.elements.rcwa.stack import (
    RCWAStack,
    RCWAYAverageWarning,
    _warn_if_shapes_y_averaged,
)

_WL = 0.55e-6
_P = 0.6e-6                     # square cell; every geometry below is
_MX = _MY = 3                   # commensurate with the S = 8k node lattices
_NS, _NSUP, _D = 1.5, 1.0, 0.22e-6
_ORDERS, _N = _harmonic_orders_2d(_MX, _MY)
_P0 = int(np.where((_ORDERS[:, 0] == 0) & (_ORDERS[:, 1] == 0))[0][0])

_EPS_L, _EPS_Y = 6.25, 6.25 + 1.2j          # lossless / lossy shape eps
_ANG = {"normal": dict(theta=0.0, phi=0.0),
        "conical": dict(theta=np.deg2rad(25.0), phi=np.deg2rad(40.0))}

# Two fill fractions per kind (the small one is the large one halved per axis).
_GEOM = {
    "rectangle": ({"shape": "rectangle", "size": (0.075e-6, 0.15e-6)},
                  {"shape": "rectangle", "size": (0.15e-6, 0.30e-6)}),
    "disk": ({"shape": "disk", "radius": 0.09e-6},
             {"shape": "disk", "radius": 0.16e-6}),
    "ellipse": ({"shape": "ellipse", "semi_axes": (0.11e-6, 0.055e-6)},
                {"shape": "ellipse", "semi_axes": (0.22e-6, 0.11e-6)}),
}
_CENTRE = (0.3e-6, 0.3e-6)


def _shape(kind, fill=1, eps=_EPS_L, centre=_CENTRE):
    return [dict(_GEOM[kind][fill], eps=eps, center=centre)]


_COVER_CACHE = {}


def _coverage(shapes, S, sub, period):
    """Per-shape coverage maps on the NODE lattice ``arange(S) * period / S``,
    wrapped in both axes: ``1`` where the node is inside the shape, or -- for
    ``sub > 1`` -- the fraction of a ``sub x sub`` sub-grid that is (the
    area-averaged raster).  Cached, because the maps depend only on the GEOMETRY
    and so are shared across every ``eps`` / angle / polarization case."""
    key = (period, S, sub, tuple(
        (sh["shape"], tuple(np.atleast_1d(
            sh.get("size", sh.get("radius", sh.get("semi_axes")))).ravel()),
         tuple(sh.get("center", (period / 2, period / 2)))) for sh in shapes))
    hit = _COVER_CACHE.get(key)
    if hit is not None:
        return hit
    off = (np.arange(sub) + 0.5) / sub - 0.5 if sub > 1 else np.zeros(1)
    out = [np.zeros((S, S)) for _ in shapes]
    base = np.arange(S) * period / S
    for ox in off:
        for oy in off:
            xx, yy = np.meshgrid(base + ox * period / S,
                                 base + oy * period / S, indexing="ij")
            for k, sh in enumerate(shapes):
                cx, cy = sh.get("center", (period / 2, period / 2))
                dx = ((xx - cx + period / 2) % period) - period / 2
                dy = ((yy - cy + period / 2) % period) - period / 2
                if sh["shape"] == "rectangle":
                    wx, wy = sh["size"]
                    m = (np.abs(dx) < wx / 2) & (np.abs(dy) < wy / 2)
                else:
                    ax, ay = ((sh["radius"], sh["radius"])
                              if sh["shape"] == "disk" else sh["semi_axes"])
                    m = (dx / ax) ** 2 + (dy / ay) ** 2 < 1.0
                out[k] += m
    out = [c / (len(off) ** 2) for c in out]
    _COVER_CACHE[key] = out
    return out


def _raster(bg, shapes, S, sub=1, period=_P):
    """One rasterised cell from :func:`_coverage`.  The shape contributions are
    SUMMED, which reproduces painting in order exactly for the DISJOINT lists
    used here (and every list rastered in this module is disjoint -- an
    overlapping one is what the W8-A guard rejects)."""
    cell = np.full((S, S), complex(bg), dtype=complex)
    for sh, cov in zip(shapes, _coverage(shapes, S, sub, period)):
        cell += (complex(sh["eps"]) - complex(bg)) * cov
    return cell


def _analytic(bg, shapes, period=_P):
    return _analytic_convolutions_2d(bg, shapes, _ORDERS, _MX, _MY,
                                     period, period)


def _solve_shapes(bg, shapes, period=_P, **kw):
    kw.setdefault("n_orders_x", _MX)
    kw.setdefault("n_orders_y", _MY)
    return rcwa_efficiency_2d_shapes(period, period, bg, shapes, _NS, _NSUP,
                                     _D, _WL, **kw)


def _solve_cell(cell, period=_P, **kw):
    kw.setdefault("n_orders_x", _MX)
    kw.setdefault("n_orders_y", _MY)
    kw.setdefault("symmetry", False)
    return rcwa_efficiency_2d(period, period, cell, _NS, _NSUP, _D, _WL, **kw)


def _rt_err(a, b):
    return max(float(np.max(np.abs(a[1] - b[1]))),
               float(np.max(np.abs(a[2] - b[2]))))


# ===========================================================================
# 1.  Oracle-free analytic identities
# ===========================================================================

@pytest.mark.parametrize("kind,area", [
    ("rectangle", 0.15e-6 * 0.30e-6),
    ("disk", np.pi * 0.16e-6 ** 2),
    ("ellipse", np.pi * 0.22e-6 * 0.11e-6),
])
def test_w8_dc_coefficient_is_the_exact_area_fraction(kind, area):
    """The sharpest single check of the form factors: ``F(0)`` must be the
    CLOSED-FORM area fill fraction, which is exact for all three kinds.
    Measured 0.0 (bit-exact) for every kind, so the pin is 1e-16 -- and the
    DC entry of ``[[eps]]`` is the exact cell-average permittivity."""
    zero = np.zeros(1)
    f_exact = area / (_P * _P)
    for eps in (_EPS_L, _EPS_Y):
        sh = _shape(kind, eps=eps)
        F0 = complex(_shape_form_factor(sh[0], zero, zero, _P, _P)[0])
        assert abs(F0 - f_exact) < 1e-16
        assert abs(F0.imag) == 0.0           # a real geometry at G = 0
        EPS, INV = _analytic(1.0, sh)
        assert abs(EPS[_P0, _P0] - (1.0 + (eps - 1.0) * f_exact)) < 1e-15
        assert abs(INV[_P0, _P0] - (1.0 + (1.0 / eps - 1.0) * f_exact)) < 1e-16


def test_w8_cell_filling_rectangle_reproduces_the_uniform_cell():
    """A rectangle covering the WHOLE cell must reproduce the uniform-cell
    answer: ``sinc(k)`` kills every ``G != 0`` coefficient exactly, leaving
    ``eps_shape`` on the diagonal.  Measured 7.3e-16 in ``[[eps]]`` and 4.4e-34
    in R/T -- and unchanged when the filler is placed OFF-centre, because the
    wrapped tiling of a cell-sized rectangle is still the full cell."""
    eps_s = 4.0 + 0.3j
    unif_E, unif_I = _analytic(eps_s, [])
    unif_rt = _solve_shapes(eps_s, [])
    for centre in ((0.3e-6, 0.3e-6), (0.13e-6, 0.44e-6)):
        full = [{"shape": "rectangle", "eps": eps_s, "size": (_P, _P),
                 "center": centre}]
        E, I = _analytic(1.0, full)
        assert np.max(np.abs(E - unif_E)) < 1e-14
        assert np.max(np.abs(I - unif_I)) < 1e-15
        assert _rt_err(_solve_shapes(1.0, full), unif_rt) < 1e-13


def test_w8_disjoint_shapes_equal_the_merged_shape():
    """Two abutting rectangles of equal ``eps`` must equal the single merged
    rectangle -- the phase convention and the additive rule, checked where the
    union IS expressible as one supported shape.  Measured 1.1e-16 (x-split),
    1.7e-16 (y-split) in ``[[eps]]`` and 5.9e-15 in R/T."""
    w, h, eps = 0.1e-6, 0.3e-6, 3.5
    for axis in (0, 1):
        def _r(cx, cy, sx, sy):
            c = (cx, cy) if axis == 0 else (cy, cx)
            s = (sx, sy) if axis == 0 else (sy, sx)
            return {"shape": "rectangle", "eps": eps, "size": s, "center": c}
        split = [_r(0.2e-6, 0.25e-6, w, h), _r(0.3e-6, 0.25e-6, w, h)]
        merged = [_r(0.25e-6, 0.25e-6, 2 * w, h)]
        Es, Is = _analytic(1.0, split)
        Em, Im = _analytic(1.0, merged)
        assert np.max(np.abs(Es - Em)) < 1e-15
        assert np.max(np.abs(Is - Im)) < 1e-15
        assert _rt_err(_solve_shapes(1.0, split),
                       _solve_shapes(1.0, merged)) < 1e-13


def test_w8_vanishing_shape_tends_to_the_background():
    """A shrinking shape must go to the pure background, and at the AREA rate:
    the form factor is ``O(r^2)`` everywhere, so a decade of radius is two
    decades of deviation.  Measured ratio 100.00 per decade over r = 1e-9 ..
    1e-13 m, and 6.4e-12 in R/T for r = 1e-12 m against an empty shape list."""
    bgE, _ = _analytic(2.25, [])
    prev = None
    for r in (1e-9, 1e-10, 1e-11, 1e-12, 1e-13):
        E, _ = _analytic(2.25, [{"shape": "disk", "eps": 12.0, "radius": r,
                                 "center": (0.3e-6, 0.2e-6)}])
        d = float(np.max(np.abs(E - bgE)))
        if prev is not None:
            assert 95.0 < prev / d < 105.0
        prev = d
    assert prev < 1e-11
    tiny = [{"shape": "disk", "eps": 12.0, "radius": 1e-12,
             "center": (0.3e-6, 0.2e-6)}]
    assert _rt_err(_solve_shapes(1.0, tiny), _solve_shapes(1.0, [])) < 1e-10
    # ... and a shape that has actually vanished is a named error, not a no-op.
    with pytest.raises(ValueError, match="non-positive dimension"):
        _solve_shapes(1.0, [{"shape": "disk", "eps": 12.0, "radius": 0.0}])


@pytest.mark.parametrize("kind", ["rectangle", "disk", "ellipse"])
def test_w8_form_factor_conjugate_symmetry(kind):
    """Real geometry ==> ``F(-G) = conj(F(G))`` (measured 0.0), hence ``[[eps]]``
    is HERMITIAN for real ``eps`` (measured 0.0) and, for complex ``eps``,
    obeys the exact loss-sign statement ``[[eps]](conj eps) = conj([[eps]])^T``
    (measured 0.0) -- the analytic path's counterpart of a real cell's
    conjugate-symmetric FFT."""
    gx = np.array([1.0, -2.0, 3.0, 0.0]) * 2 * np.pi / _P
    gy = np.array([-1.0, 2.0, 0.0, 0.0]) * 2 * np.pi / _P
    sh = _shape(kind)[0]
    Fp = _shape_form_factor(sh, gx, gy, _P, _P)
    Fm = _shape_form_factor(sh, -gx, -gy, _P, _P)
    assert np.max(np.abs(Fm - np.conj(Fp))) < 1e-17
    off = (0.21e-6, 0.17e-6)               # off-centre: a non-trivial phase
    E, I = _analytic(1.0, _shape(kind, centre=off))
    assert np.max(np.abs(E - E.conj().T)) < 1e-16
    assert np.max(np.abs(I - I.conj().T)) < 1e-16
    El, _ = _analytic(1.0 + 0.05j, _shape(kind, eps=_EPS_Y, centre=off))
    Ec, _ = _analytic(1.0 - 0.05j,
                      _shape(kind, eps=np.conj(_EPS_Y), centre=off))
    assert np.max(np.abs(Ec - El.conj().T)) < 1e-16


# ===========================================================================
# 2.  Convergence cross-validation: the raster must converge TO the analytic
# ===========================================================================

@pytest.mark.parametrize("kind", ["rectangle", "disk", "ellipse"])
@pytest.mark.parametrize("fill", [0, 1])
@pytest.mark.parametrize("eps", [_EPS_L, _EPS_Y], ids=["lossless", "lossy"])
def test_w8_operator_convergence_area_averaged(kind, fill, eps):
    """``[[eps]]`` from an AREA-AVERAGED raster converges to the analytic
    spectrum at second order: each doubling QUARTERS the gap (the strong form
    of "refinement halves the gap").  Measured per-step ratios 3.46 .. 4.55
    over S = 64 -> 128 -> 256 across all 12 (kind, fill, eps) combinations --
    exactly 4.00 for the rectangle, whose edges land on every node lattice
    here -- so the pin is 3.0 per step, with the S = 256 gap under 4e-4."""
    sh = _shape(kind, fill, eps)
    EPSa, _ = _analytic(1.0, sh)
    err = [float(np.max(np.abs(
        _eps_convolution_2d(_raster(1.0, sh, S, 16), _ORDERS, _MX, _MY)
        - EPSa))) for S in (64, 128, 256)]
    assert err[0] < 6e-3
    for a, b in zip(err, err[1:]):
        assert b < a / 3.0
    assert err[-1] < 4e-4


@pytest.mark.parametrize("kind", ["rectangle", "disk", "ellipse"])
@pytest.mark.parametrize("eps", [_EPS_L, _EPS_Y], ids=["lossless", "lossy"])
def test_w8_operator_convergence_point_sampled(kind, eps):
    """The NAIVE point-sampled raster -- what a caller writes by hard-assigning
    pixels -- also converges to the analytic spectrum, but only at FIRST order,
    and NOT monotonically per step: the realised feature width is quantised to
    the node lattice, so the gap sawtooths with ``S`` (measured per-step ratios
    as low as 1.58 for the ellipse).  Over a 4x refinement the sawtooth closes:
    net 3.97 (rectangle -- exactly the 1.99 + 1.99 of ``O(1/S)``), 9.52 (disk),
    4.50 (ellipse).  This is the honest cost of rasterising, and the reason the
    analytic path exists."""
    sh = _shape(kind, 1, eps)
    EPSa, _ = _analytic(1.0, sh)
    err = [float(np.max(np.abs(
        _eps_convolution_2d(_raster(1.0, sh, S), _ORDERS, _MX, _MY) - EPSa)))
        for S in (128, 256, 512)]
    assert err[0] / err[-1] > 3.5              # net, over the 4x refinement
    assert err[-1] < 1e-2
    if kind == "rectangle":
        for a, b in zip(err, err[1:]):         # sharp O(1/S), two-sided
            assert 1.8 < a / b < 2.2


@pytest.mark.parametrize("kind", ["rectangle", "disk", "ellipse"])
@pytest.mark.parametrize("eps", [_EPS_L, _EPS_Y], ids=["lossless", "lossy"])
@pytest.mark.parametrize("ang", ["normal", "conical"])
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_w8_efficiency_convergence_to_the_analytic_answer(kind, eps, ang, pol):
    """The user-facing statement: R/T from the rasterised path converge TO the
    analytic-shape answer, over all 24 (kind, eps, angle, polarization)
    combinations, with no systematic residual.

    The RECTANGLE is the clean instrument -- its edges lie on every node
    lattice used here, so the area-averaged raster is second-order with no
    sawtooth and each doubling QUARTERS the R/T gap (measured 3.98 .. 4.00 per
    step for all 8 of its combinations, net 15.9 .. 16.0).  A curved wall
    (disk / ellipse) keeps a residual sub-pixel boundary term at ``sub = 8``,
    so its per-step ratio is not monotone (measured down to 0.90) while the net
    over the 4x refinement stays >= 4.9; hence the net form there.  Envelope at
    S = 256: 3.8e-5 worst-case over all combinations."""
    sh = _shape(kind, 1, eps)
    ref = _solve_shapes(1.0, sh, polarization=pol, **_ANG[ang])
    err = [_rt_err(ref, _solve_cell(_raster(1.0, sh, S, 8),
                                    polarization=pol, **_ANG[ang]))
           for S in (64, 128, 256)]
    assert err[0] < 3e-3
    assert err[-1] < 1e-4
    assert err[0] / err[-1] > 3.5
    if kind == "rectangle":
        for a, b in zip(err, err[1:]):
            assert b < a / 3.0


@pytest.mark.parametrize("kind", ["rectangle", "disk", "ellipse"])
@pytest.mark.parametrize("eps,ang,pol", [(_EPS_L, "normal", "te"),
                                         (_EPS_Y, "conical", "tm")],
                         ids=["lossless-normal-te", "lossy-conical-tm"])
def test_w8_efficiency_convergence_at_the_small_fill(kind, eps, ang, pol):
    """The same statement at the SECOND fill fraction (each axis halved, so a
    4x smaller feature), which the operator-level test covers for all 12
    combinations and this samples at the efficiency level: measured net 5.60 ..
    29.95 over the 4x refinement, reaching 5.0e-6 .. 6.0e-5 at S = 256."""
    sh = _shape(kind, 0, eps)
    ref = _solve_shapes(1.0, sh, polarization=pol, **_ANG[ang])
    err = [_rt_err(ref, _solve_cell(_raster(1.0, sh, S, 8),
                                    polarization=pol, **_ANG[ang]))
           for S in (64, 256)]
    assert err[0] < 3e-3
    assert err[-1] < 1e-4
    assert err[0] / err[-1] > 3.5


def test_w8_disjoint_pair_matches_the_painted_raster():
    """Superposition and paint-in-order agree on a DISJOINT list -- the premise
    the overlap guard enforces -- so the two-disk geometry of audit M1 must
    converge to its painted raster: measured 1.31e-4 -> 5.80e-5 -> 8.40e-6 for
    S = 64 -> 128 -> 256 (net 15.6) on the 0.9 um cell."""
    p9 = 0.9e-6
    sh = [{"shape": "disk", "radius": 0.20e-6, "eps": 6.0,
           "center": (0.225e-6, 0.225e-6)},
          {"shape": "disk", "radius": 0.10e-6, "eps": 6.0,
           "center": (0.63e-6, 0.54e-6)}]
    ref = _solve_shapes(1.0, sh, period=p9)
    err = [_rt_err(ref, _solve_cell(_raster(1.0, sh, S, 8, period=p9),
                                    period=p9)) for S in (64, 256)]
    assert err[0] < 1e-3
    assert err[-1] < 3e-5
    assert err[0] / err[-1] > 3.5


def test_w8_inverse_eps_second_return_is_the_analytic_1_over_eps():
    """``_analytic_convolutions_2d``'s second return is documented as the
    DIRECT-rule Laurent transform of ``1/eps`` (kept for factorization studies,
    read by no shipped formulation since audit M10).  Cross-validate the claim:
    it must be what an FFT of a ``1/eps`` cell converges to -- measured net
    3.73 .. 4.28 over a 4x refinement for all three kinds, lossless and lossy,
    reaching 1.8e-3 at S = 256."""
    for kind in ("rectangle", "disk", "ellipse"):
        for eps in (_EPS_L, _EPS_Y):
            sh = _shape(kind, 1, eps)
            _E, INVa = _analytic(1.0, sh)
            err = [float(np.max(np.abs(
                _eps_convolution_2d(1.0 / _raster(1.0, sh, S, 16), _ORDERS,
                                    _MX, _MY) - INVa))) for S in (64, 256)]
            assert err[0] / err[-1] > 3.0
            assert err[-1] < 3e-3


# ===========================================================================
# 3.  Off-centre and periodic-wrap geometry
# ===========================================================================

@pytest.mark.parametrize("kind", ["rectangle", "disk", "ellipse"])
def test_w8_periodic_wrap_is_exact_not_asymptotic(kind):
    """A shape running off a cell edge continues PERIODICALLY, and the analytic
    form factor gets that exactly right for free: it is sampled on the
    reciprocal lattice, where the transform of one shape and the transform of
    its whole wrapped tiling coincide (Poisson).  So moving a shape -- to a
    corner, across a seam, or outside the cell entirely -- must leave
    ``|[[eps]]|`` invariant (measured <= 1.1e-16) and, at normal incidence,
    leave R/T invariant (measured 1.1e-14): a pure translation is a pure phase
    on every coefficient."""
    ref_E, _ = _analytic(1.0, _shape(kind))
    ref_rt = _solve_shapes(1.0, _shape(kind))
    for centre in [(0.0, 0.0), (0.02e-6, 0.58e-6), (_P, _P),
                   (-0.3e-6, 0.9e-6)]:
        E, _ = _analytic(1.0, _shape(kind, centre=centre))
        assert np.max(np.abs(np.abs(E) - np.abs(ref_E))) < 1e-14
        assert abs(E[_P0, _P0] - ref_E[_P0, _P0]) < 1e-16      # DC is real
        assert _rt_err(_solve_shapes(1.0, _shape(kind, centre=centre)),
                       ref_rt) < 1e-12


@pytest.mark.parametrize("kind", ["rectangle", "disk", "ellipse"])
def test_w8_wrapped_shape_matches_a_wrapped_raster(kind):
    """The wrap is not just self-consistent -- it agrees with the ORACLE.  A
    shape centred on the cell corner, rasterised with the same wrap, converges
    to the analytic answer with the same gap as the centred shape does
    (measured bit-identical, |difference| = 0.0, at every S: the geometry is a
    lattice translation of the centred one)."""
    for eps in (_EPS_L, _EPS_Y):
        centred, corner = _shape(kind, 1, eps), _shape(kind, 1, eps, (0.0, 0.0))
        Ec, _ = _analytic(1.0, centred)
        Ew, _ = _analytic(1.0, corner)
        err = []
        for S in (64, 256):
            err.append((
                float(np.max(np.abs(_eps_convolution_2d(
                    _raster(1.0, centred, S, 16), _ORDERS, _MX, _MY) - Ec))),
                float(np.max(np.abs(_eps_convolution_2d(
                    _raster(1.0, corner, S, 16), _ORDERS, _MX, _MY) - Ew)))))
        for centre_err, wrap_err in err:
            assert abs(wrap_err - centre_err) < 1e-15
        assert err[0][1] / err[1][1] > 3.0       # and it still converges


# ===========================================================================
# 4.  W8-A: the additive layering rule, documented and enforced
# ===========================================================================

_OVERLAPPING = {
    "two rectangles, 5/6 overlap":
        [{"shape": "rectangle", "eps": 4.0, "size": (0.30e-6, 0.20e-6),
          "center": (0.25e-6, 0.30e-6)},
         {"shape": "rectangle", "eps": 9.0, "size": (0.30e-6, 0.20e-6),
          "center": (0.35e-6, 0.30e-6)}],
    "two disks, partial":
        [{"shape": "disk", "eps": _EPS_L, "radius": 0.15e-6,
          "center": (0.25e-6, 0.30e-6)},
         {"shape": "disk", "eps": _EPS_L, "radius": 0.15e-6,
          "center": (0.35e-6, 0.30e-6)}],
    "two IDENTICAL disks":
        [{"shape": "disk", "eps": _EPS_L, "radius": 0.15e-6,
          "center": (0.30e-6, 0.30e-6)}] * 2,
    "a cross of two rectangles":
        [{"shape": "rectangle", "eps": 4.0, "size": (0.40e-6, 0.10e-6),
          "center": (0.30e-6, 0.30e-6)},
         {"shape": "rectangle", "eps": 4.0, "size": (0.10e-6, 0.40e-6),
          "center": (0.30e-6, 0.30e-6)}],
    "a disk inside a rectangle":
        [{"shape": "rectangle", "eps": 4.0, "size": (0.40e-6, 0.40e-6),
          "center": (0.30e-6, 0.30e-6)},
         {"shape": "disk", "eps": 9.0, "radius": 0.05e-6,
          "center": (0.30e-6, 0.30e-6)}],
    "an ellipse crossing a disk":
        [{"shape": "ellipse", "eps": 4.0, "semi_axes": (0.25e-6, 0.06e-6),
          "center": (0.30e-6, 0.30e-6)},
         {"shape": "disk", "eps": 9.0, "radius": 0.08e-6,
          "center": (0.30e-6, 0.36e-6)}],
    "an overlap only ACROSS the periodic seam":
        [{"shape": "disk", "eps": _EPS_L, "radius": 0.12e-6,
          "center": (0.02e-6, 0.30e-6)},
         {"shape": "disk", "eps": _EPS_L, "radius": 0.12e-6,
          "center": (0.58e-6, 0.30e-6)}],
}

_DISJOINT = {
    "two disks, 1 pm apart":
        [{"shape": "disk", "eps": _EPS_L, "radius": 0.10e-6,
          "center": (0.15e-6, 0.30e-6)},
         {"shape": "disk", "eps": _EPS_L, "radius": 0.10e-6,
          "center": (0.350001e-6, 0.30e-6)}],
    "two disks EXACTLY tangent":
        [{"shape": "disk", "eps": _EPS_L, "radius": 0.10e-6,
          "center": (0.15e-6, 0.30e-6)},
         {"shape": "disk", "eps": _EPS_L, "radius": 0.10e-6,
          "center": (0.35e-6, 0.30e-6)}],
    "two rectangles EXACTLY abutting":
        [{"shape": "rectangle", "eps": _EPS_L, "size": (0.2e-6, 0.3e-6),
          "center": (0.2e-6, 0.3e-6)},
         {"shape": "rectangle", "eps": _EPS_L, "size": (0.2e-6, 0.3e-6),
          "center": (0.4e-6, 0.3e-6)}],
    "two diagonal disks whose BOXES overlap":
        [{"shape": "disk", "eps": _EPS_L, "radius": 0.14e-6,
          "center": (0.12e-6, 0.12e-6)},
         {"shape": "disk", "eps": _EPS_L, "radius": 0.14e-6,
          "center": (0.33e-6, 0.33e-6)}],
    "a diagonal disk and rectangle whose BOXES overlap":
        [{"shape": "rectangle", "eps": _EPS_L, "size": (0.2e-6, 0.2e-6),
          "center": (0.14e-6, 0.14e-6)},
         {"shape": "disk", "eps": _EPS_L, "radius": 0.1e-6,
          "center": (0.36e-6, 0.36e-6)}],
    "a 2x2 lattice of disks":
        [{"shape": "disk", "eps": _EPS_L, "radius": 0.145e-6,
          "center": (cx, cy)}
         for cx in (0.15e-6, 0.45e-6) for cy in (0.15e-6, 0.45e-6)],
    "a wrapped disk beside a centred one":
        [{"shape": "disk", "eps": _EPS_L, "radius": 0.14e-6,
          "center": (0.0, 0.0)},
         {"shape": "disk", "eps": _EPS_L, "radius": 0.14e-6,
          "center": (0.3e-6, 0.3e-6)}],
}


@pytest.mark.parametrize("name", list(_OVERLAPPING))
def test_w8a_overlapping_shapes_are_rejected(name):
    """W8-A.  The analytic factorization ADDS form factors, so an overlap gets
    ``eps_bg + (eps_1 - eps_bg) + (eps_2 - eps_bg)`` on the shared area --
    neither shape's ``eps`` -- while conserving energy perfectly.  Before the
    guard every list here was accepted: the two rectangles moved R/T by 6.1e-2
    (DC permittivity 2.833 against the painted 2.501), the partial disks by
    1.3e-1, the identical disks by 1.1e-1 (DC 3.062 against 2.031), closures
    -6.7e-16 / -3.8e-15 / -1.9e-14.  Every case also passes the per-shape AND
    cumulative-area guards, so nothing else catches them."""
    with pytest.raises(ValueError, match="OVERLAP"):
        _solve_shapes(1.0, _OVERLAPPING[name])
    with pytest.raises(ValueError, match="add_layer:.*OVERLAP"):
        RCWAStack(_P, period_y=_P, n_orders=_MX, n_orders_y=_MY).add_layer(
            _D, shapes=_OVERLAPPING[name], eps_background=1.0)


@pytest.mark.parametrize("name", list(_DISJOINT))
def test_w8a_disjoint_shapes_are_not_rejected(name):
    """The other half of W8-A: the guard must not cost any legal geometry.
    Exactly TANGENT circles and exactly ABUTTING rectangles are legal (their
    intersection has measure zero, so superposition is still right), and so are
    diagonal neighbours whose BOUNDING BOXES overlap -- the box test is only a
    pre-filter, the verdict comes from the support-function separating axis.
    All seven lists solve with a <= 3e-14 energy closure."""
    _o, R, T = _solve_shapes(1.0, _DISJOINT[name])
    assert abs(float(R.sum() + T.sum()) - 1.0) < 3e-14


def test_w8a_overlap_predicate_is_one_sided_at_tangency():
    """``_shapes_overlap`` is deliberately one-sided: with a ``1e-6 * period``
    slack (6e-13 m here) and a 4096-direction support scan whose worst
    under-estimate is ~2e-7 * period, a pair that TOUCHES or is gapped is never
    called overlapping, while an overlap deeper than ~1e-6 of a period always
    is.  Measured switch-over between 1e-13 m (not reported) and 1e-11 m
    (reported) of overlap depth, identically for disk-disk, ellipse-ellipse and
    rectangle-ellipse."""
    r = 0.08e-6
    for gap, expect in ((-1e-8, True), (-1e-9, True), (-1e-11, True),
                        (-1e-13, False), (0.0, False), (1e-13, False),
                        (1e-11, False), (1e-9, False)):
        a = ("disk", r, r, (0.15e-6, 0.3e-6))
        b = ("disk", r, r, (0.15e-6 + 2 * r + gap, 0.3e-6))
        ea = ("ellipse", 0.10e-6, r, (0.15e-6, 0.3e-6))
        eb = ("ellipse", r, 0.05e-6, (0.15e-6 + 0.10e-6 + r + gap, 0.3e-6))
        ra = ("rectangle", 0.10e-6, r, (0.15e-6, 0.3e-6))
        assert _shapes_overlap(a, b, _P, _P) is expect
        assert _shapes_overlap(ea, eb, _P, _P) is expect
        assert _shapes_overlap(ra, eb, _P, _P) is expect
        assert _shapes_overlap(b, a, _P, _P) is expect          # symmetric


def test_w8a_the_guard_reaches_the_DISPERSIVE_shapes_route():
    """``RCWAStack.add_layer`` defers validation when any ``eps`` is a
    ``wl ->`` callable (there is nothing to check yet), so confirm the overlap
    still cannot reach a solve: the dispersive shapes list is validated once
    materialised, and ``solve_vs_wavelength`` is the only way to solve a
    dispersive stack (``solve`` refuses it outright)."""
    ov = [dict(sh, eps=(lambda wl: _EPS_L))
          for sh in _OVERLAPPING["two disks, partial"]]
    st = RCWAStack(_P, period_y=_P, n_orders=_MX, n_orders_y=_MY)
    st.add_layer(_D, shapes=ov, eps_background=1.0)          # deferred
    with pytest.raises(ValueError, match="solve_vs_wavelength:.*OVERLAP"):
        st.set_source(_WL).solve_vs_wavelength([_WL, _WL * 1.01])


def test_w8a_the_docstring_no_longer_promises_paint_over():
    """The defect was a CONTRACT mismatch, so pin the contract: the entry
    point's docstring must state the additive rule and the disjointness
    requirement, and must not still claim shapes are "painted in order"."""
    doc = rcwa_efficiency_2d_shapes.__doc__
    assert "painted in order" not in doc
    assert "ADDED" in doc and "DISJOINT" in doc
    assert "rcwa_efficiency_2d" in doc          # where paint-over does live


# ===========================================================================
# 5.  W8-B: n_orders_y = 0 needs a y-invariant shape list
# ===========================================================================

def test_w8b_n_orders_y0_rejects_a_y_varying_shape_list():
    """W8-B.  With no retained y-harmonic only the y-AVERAGED permittivity
    enters, so a y-varying shape list silently solves a DIFFERENT structure:
    measured R00 = 0.054846364 for a disk against the y-resolved 0.006896833
    (8x), and bit-comparable to the explicitly y-averaged pixel cell
    (0.054848266, the raster's own residual), with a 4.4e-16 energy closure --
    no tripwire can see it.  The pixel path has rejected exactly this since
    audit M8; the analytic path now matches, on both entry points."""
    disk = _shape("disk")
    with pytest.raises(ValueError, match="y-INVARIANT shape list"):
        _solve_shapes(1.0, disk, n_orders_x=6, n_orders_y=0)
    with pytest.raises(ValueError, match="add_layer:.*y-INVARIANT shape list"):
        RCWAStack(_P, period_y=_P, n_orders=6, n_orders_y=0).add_layer(
            _D, shapes=disk, eps_background=1.0)
    for kind in ("disk", "ellipse"):
        with pytest.raises(ValueError, match="y-INVARIANT shape list"):
            _solve_shapes(1.0, _shape(kind), n_orders_x=6, n_orders_y=0)
    # A rectangle that does not span period_y is y-varying too.
    with pytest.raises(ValueError, match="y-INVARIANT shape list"):
        _solve_shapes(1.0, _shape("rectangle"), n_orders_x=6, n_orders_y=0)


def test_w8b_n_orders_y0_still_accepts_a_y_invariant_list():
    """The M8 fast path must survive: a rectangle spanning the FULL period_y is
    a stripe, exactly y-invariant, and ``n_orders_y = 0`` is then free -- it
    reproduces the y-resolved solve to 2.9e-16 (measured, R00 = 0.066137988037
    either way) at 1/3 of the retained harmonics.  An empty shape list (uniform
    slab) is y-invariant too, and a 1-D ``RCWAStack``, whose ``noy = 0`` is a
    sentinel and not a truncation choice, is deliberately left alone."""
    stripe = [{"shape": "rectangle", "eps": _EPS_L, "size": (0.24e-6, _P),
               "center": (0.3e-6, 0.3e-6)}]
    o0, R0, T0 = _solve_shapes(1.0, stripe, n_orders_x=6, n_orders_y=0)
    o2, R2, T2 = _solve_shapes(1.0, stripe, n_orders_x=6, n_orders_y=1)
    i0 = int(np.where((o0[:, 0] == 0) & (o0[:, 1] == 0))[0][0])
    i2 = int(np.where((o2[:, 0] == 0) & (o2[:, 1] == 0))[0][0])
    assert abs(float(R0[i0] - R2[i2])) < 1e-13
    assert abs(float(R0.sum() + T0.sum()) - 1.0) < 1e-13
    _o, R, T = _solve_shapes(2.25, [], n_orders_x=6, n_orders_y=0)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-13
    st = RCWAStack(_P, n_orders=4)                     # 1-D stack: noy = 0
    assert st.is_1d and st.noy == 0
    with pytest.warns(RCWAYAverageWarning):            # diagnosed, not rejected
        st.add_layer(_D, shapes=_shape("disk"), eps_background=1.0)


# ===========================================================================
# 5b.  W8-B': the 1-D-stack half -- a DIAGNOSTIC, not a rejection
# ===========================================================================
#
# A 1-D (mono-periodic) RCWAStack carries only the n = 0 y-harmonic, so its
# noy = 0 is a SENTINEL ("I am 1-D"), not a truncation choice -- the strict
# raise above must not fire there, exactly as on the pixel path.  What was
# missing is the other half of the pixel path's treatment: the
# RCWAYAverageWarning diagnostic that commit 809314c added for a y-VARYING
# eps_cell had no analytic-shape flavour, so a disk on a 1-D stack solved its
# y-AVERAGE in complete silence.  Measured through the stack API (P = 0.6 um,
# lambda = 550 nm, d = 220 nm, eps 6.25 disk r = 160 nm, n_orders = 6):
#
#   1-D stack + disk shapes     R00 = 0.054846364   closure -8.9e-16
#   2-D stack + the same disk   R00 = 0.006896833   closure -1.5e-14
#                               ratio 7.95x, absolute error 0.047950
#
# and the mechanism is exactly y-averaging: feeding the explicitly y-AVERAGED
# pixel cell of the same disk to the same 1-D stack gives 0.054845052 at
# S = 512 and 0.054846287 at S = 2048 -- converging on the shapes answer
# (|d| = 1.3e-06 -> 7.7e-08, the raster's own residual).

_Y_VARIANCE_CASES = [
    ("disk", _shape("disk"), True),
    ("ellipse", _shape("ellipse"), True),
    ("short rectangle", _shape("rectangle"), True),
    ("full-height rectangle",
     [{"shape": "rectangle", "eps": _EPS_L, "size": (0.24e-6, _P),
       "center": (0.3e-6, 0.3e-6)}], False),
    ("stripe + disk",
     [{"shape": "rectangle", "eps": _EPS_L, "size": (0.1e-6, _P),
       "center": (0.1e-6, 0.3e-6)},
      {"shape": "disk", "eps": _EPS_L, "radius": 0.1e-6,
       "center": (0.4e-6, 0.3e-6)}], True),
    ("empty list", [], False),
]


@pytest.mark.parametrize("name,shapes,y_varying",
                         [(n, s, v) for n, s, v in _Y_VARIANCE_CASES],
                         ids=[n for n, _s, _v in _Y_VARIANCE_CASES])
def test_w8b_one_d_stack_warns_exactly_when_the_layer_varies_in_y(
        name, shapes, y_varying):
    """The diagnostic fires on a y-VARYING shape list on a 1-D stack and stays
    silent on a y-INVARIANT one -- a full-``period_y`` rectangle (an x-only
    grating expressed analytically, the legitimate idiom) and an empty list
    both lose nothing to the average.  A list that MIXES them warns, naming the
    offending index (the disk at ``shapes[1]``, not the stripe at
    ``shapes[0]``)."""
    st = RCWAStack(_P, n_orders=4, n_substrate=1.5)
    assert st.is_1d and st.noy == 0
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        st.add_layer(_D, shapes=shapes, eps_background=1.0)
    hits = [w for w in rec if w.category is RCWAYAverageWarning]
    assert len(hits) == int(y_varying), (
        f"{name}: expected warn={y_varying}; got "
        f"{[w.category.__name__ for w in rec]}")
    if y_varying:
        msg = str(hits[0].message)
        assert "y-AVERAGED" in msg
        assert "period_y" in msg and "n_orders_y" in msg   # names the fix
        assert f"shapes[{_shapes_y_varying(shapes, _P)}]" in msg


@pytest.mark.parametrize("name,shapes,y_varying",
                         [(n, s, v) for n, s, v in _Y_VARIANCE_CASES],
                         ids=[n for n, _s, _v in _Y_VARIANCE_CASES])
def test_w8b_explicit_2d_stack_never_warns(name, shapes, y_varying):
    """An ``n_orders_y >= 1`` stack RESOLVES y, so there is nothing to report;
    and with ``n_orders_y = 0`` the SAME input is a hard error instead (the
    truncation was a deliberate choice there) -- the two halves of the contract
    never overlap."""
    st = RCWAStack(_P, period_y=_P, n_orders=4, n_orders_y=4, n_substrate=1.5)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        st.add_layer(_D, shapes=shapes, eps_background=1.0)
    assert not [w for w in rec if w.category is RCWAYAverageWarning]
    st0 = RCWAStack(_P, period_y=_P, n_orders=4, n_orders_y=0, n_substrate=1.5)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        if y_varying:
            with pytest.raises(ValueError, match="y-INVARIANT shape list"):
                st0.add_layer(_D, shapes=shapes, eps_background=1.0)
        else:
            st0.add_layer(_D, shapes=shapes, eps_background=1.0)
    assert not [w for w in rec if w.category is RCWAYAverageWarning], (
        "the explicitly-2-D path raises; it must not ALSO warn")


@pytest.mark.parametrize("name,shapes,y_varying",
                         [(n, s, v) for n, s, v in _Y_VARIANCE_CASES],
                         ids=[n for n, _s, _v in _Y_VARIANCE_CASES])
def test_w8b_warning_verdict_matches_the_strict_raise(name, shapes, y_varying):
    """The no-divergence contract commit 809314c pinned for the pixel path
    (``_warn_if_y_averaged`` vs the M8 ``strict_y`` branch), carried over to the
    analytic flavour: the 1-D-stack WARNING and the 2-D ``n_orders_y = 0`` RAISE
    read the SAME predicate (:func:`_shapes_y_varying`), so a future edit to one
    cannot silently disagree with the other about the same shape list."""
    raised = False
    try:
        _validate_shapes("probe", shapes, _P, _P, n_orders_y=0)
    except ValueError as e:
        raised = "y-INVARIANT shape list" in str(e)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warned = _warn_if_shapes_y_averaged("probe", shapes, _P, 0,
                                            strict_y=False)
    assert warned is bool(warned)
    assert warned == y_varying
    assert raised == warned, (
        "the 1-D warning and the 2-D n_orders_y=0 raise must reach the SAME "
        f"verdict about y-variance; raise={raised} warn={warned}")
    assert (_shapes_y_varying(shapes, _P) is not None) == y_varying
    assert len([w for w in rec
                if w.category is RCWAYAverageWarning]) == int(warned)


def test_w8b_shapes_warning_helper_noops_like_its_pixel_twin():
    """``strict_y`` on means the raise already fired -- no double report; and a
    nonzero ``n_orders_y`` means nothing is averaged away."""
    disk = _shape("disk")
    assert _warn_if_shapes_y_averaged("p", disk, _P, 0, strict_y=True) is False
    assert _warn_if_shapes_y_averaged("p", disk, _P, 4, strict_y=False) is False


def test_w8b_shapes_warning_is_a_diagnostic_not_a_rejection():
    """The 1-D + shapes contract is UNCHANGED: the solve still runs and still
    returns the y-AVERAGED answer.  Measured R00 = 0.054846364 on the 1-D stack
    against the y-resolved 0.006896833 (7.95x, absolute 0.047950), both with
    ~1e-15 closures -- and the 1-D answer tracks the explicitly y-AVERAGED
    PIXEL cell of the same disk (0.054845052 at S = 512, |d| = 1.3e-06; 7.7e-08
    at S = 2048), which is what makes "it y-averages" a measurement rather than
    an inference."""
    disk = _shape("disk")

    def _R00(st):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RCWAYAverageWarning)
            res = st.set_source(_WL).solve()
        o, R, T = (np.asarray(v) for v in res.efficiencies())
        i = (int(np.where(o == 0)[0][0]) if o.ndim == 1 else
             int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0]))
        return float(R[1][i]), float(R[1].sum() + T[1].sum() - 1.0)

    st1 = RCWAStack(_P, n_orders=6, n_substrate=_NS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RCWAYAverageWarning)
        st1.add_layer(_D, shapes=disk, eps_background=1.0)
    r1, c1 = _R00(st1)
    st2 = RCWAStack(_P, period_y=_P, n_orders=6, n_orders_y=6, n_substrate=_NS)
    st2.add_layer(_D, shapes=disk, eps_background=1.0)
    r2, c2 = _R00(st2)
    assert abs(c1) < 1e-13 and abs(c2) < 1e-13     # both energy-clean
    assert r1 / r2 > 7.0                           # the silent trap, measured
    # mechanism: the same 1-D stack fed the explicitly y-AVERAGED pixel cell
    S = 512
    cov = _coverage(disk, S, 8, _P)[0]
    cell = (1.0 + (_EPS_L - 1.0) * cov).astype(complex)
    yavg = np.repeat(cell.mean(axis=1, keepdims=True), S, axis=1)
    st3 = RCWAStack(_P, n_orders=6, n_substrate=_NS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RCWAYAverageWarning)
        st3.add_layer(_D, eps_cell=yavg)
    r3, _c3 = _R00(st3)
    assert abs(r3 - r1) < 5e-6


def test_w8b_shapes_warning_can_be_silenced_and_promoted_by_category():
    """Same category as the pixel flavour, so one filter covers both and
    neither touches the sibling physics diagnostics."""
    disk = _shape("disk")
    assert issubclass(RCWAYAverageWarning, UserWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RCWAYAverageWarning)
        with pytest.raises(RCWAYAverageWarning):
            RCWAStack(_P, n_orders=4).add_layer(_D, shapes=disk,
                                                eps_background=1.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", RCWAYAverageWarning)
        RCWAStack(_P, n_orders=4).add_layer(_D, shapes=disk,
                                            eps_background=1.0)
    assert not [w for w in rec if w.category is RCWAYAverageWarning]


def test_w8b_dispersive_shapes_on_a_1d_stack_warn_at_materialization():
    """A dispersive (``wl ->``) shape list is validated -- and now diagnosed --
    where it is materialised, one report per wavelength, exactly like the
    dispersive pixel cell: measured to surface from the SAME frame and with the
    SAME count as its pixel twin on the same sweep (both land in the threaded
    sweep's worker, the ``stacklevel`` clamp its own docstring documents), so the
    two flavours cannot drift in reporting either."""
    def _report(add):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            st = RCWAStack(_P, n_orders=4, n_substrate=_NS)
            add(st)
            st.set_source(_WL)
            st.solve_vs_wavelength([_WL, _WL * 1.01])
        hits = [w for w in rec if w.category is RCWAYAverageWarning]
        return [(w.filename, w.lineno) for w in hits]

    S = 24
    xy = np.arange(S) * _P / S
    xx, yy = np.meshgrid(xy, xy, indexing="ij")
    pillar = np.where((np.abs(xx - 0.3e-6) < 0.15e-6)
                      & (np.abs(yy - 0.3e-6) < 0.15e-6),
                      _EPS_L, 1.0).astype(complex)
    shapes_frames = _report(lambda st: st.add_layer(
        _D, shapes=[dict(_shape("disk")[0], eps=lambda wl: _EPS_L)],
        eps_background=1.0))
    pixel_frames = _report(lambda st: st.add_layer(
        _D, eps_cell=lambda wl: pillar))
    assert len(shapes_frames) == 2                  # one report per wavelength
    assert shapes_frames == pixel_frames
    stripe = [{"shape": "rectangle", "eps": lambda wl: _EPS_L,
               "size": (0.24e-6, _P), "center": (0.3e-6, 0.3e-6)}]
    st2 = RCWAStack(_P, n_orders=4, n_substrate=_NS)
    st2.add_layer(_D, shapes=stripe, eps_background=1.0)
    st2.set_source(_WL)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        st2.solve_vs_wavelength([_WL, _WL * 1.01])
    assert not [w for w in rec if w.category is RCWAYAverageWarning]


# ===========================================================================
# 6.  W8-C: malformed shape dicts are named errors
# ===========================================================================

@pytest.mark.parametrize("shapes,match", [
    ([{"shape": "disk", "eps": 6.25}], "missing the required 'radius'"),
    ([{"shape": "disk", "eps": 6.25, "semi_axes": (1e-7, 1e-7)}],
     "missing the required 'radius'"),
    ([{"shape": "rectangle", "eps": 6.25}], "missing the required 'size'"),
    ([{"shape": "ellipse", "eps": 6.25}], "missing the required 'semi_axes'"),
    ([{"shape": "disk", "radius": 1e-7}], "missing the required 'eps'"),
    ([{"shape": "disk", "radius": 1e-7, "eps": None}],
     "missing the required 'eps'"),
    ([{"shape": "rectangle", "eps": 6.25, "size": 1e-7}],
     "non-numeric 'size'"),
    ([{"shape": "disk", "eps": 6.25, "radius": 1e-7,
       "center": (1e-7, 1e-7, 1e-7)}], "'center' of length 3"),
    ([{"shape": "disk", "eps": 6.25, "radius": 1e-7,
       "center": (np.nan, 0.0)}], "non-finite center"),
    ([{"shape": "triangle", "eps": 6.25, "size": (1e-7, 1e-7)}],
     "unknown shape 'triangle'"),
    (["disk"], r"shapes\[0\] must be a dict"),
    ({"shape": "disk", "eps": 6.25, "radius": 1e-7}, "must be a LIST"),
])
def test_w8c_malformed_shape_dicts_are_named_errors(shapes, match):
    """W8-C.  Every one of these used to escape as a bare ``KeyError('radius')``
    / ``TypeError: 'float' object is not iterable`` / ``AttributeError`` from
    inside the form factors, with no function name and no hint (house rule:
    named errors)."""
    with pytest.raises(ValueError, match=match) as ei:
        _solve_shapes(1.0, shapes)
    assert str(ei.value).startswith("rcwa_efficiency_2d_shapes: ")


def test_w8c_the_unknown_kind_message_still_names_the_helper():
    """``_shape_form_factor`` is exported and callable directly; its own guard
    must keep naming itself (the public entry point's list is checked earlier
    by :func:`_validate_shapes`)."""
    with pytest.raises(ValueError, match="_shape_form_factor: unknown shape"):
        _shape_form_factor({"shape": "hexagon"}, np.zeros(1), np.zeros(1),
                           _P, _P)
