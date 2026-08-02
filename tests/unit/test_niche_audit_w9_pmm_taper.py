"""AUDIT W9 -- ``PMMStack`` tapered-builder ``shear`` (capability parity with
``RCWAStack.add_tapered_grating``) + the measured z-staircase error law.

TASK A -- ``shear`` on ``PMMStack.add_tapered_grating`` / ``add_tapered_ridges``
===============================================================================

The RCWA sibling has carried sheared (parallelogram) sidewalls since v5.14.1
(audit GAP1); the PMM twins could only build a SYMMETRIC trapezoid, so a
slanted-wall fab profile had no laterally-exact (no-Fourier-floor) solver.
``shear`` is now mirrored with the EXACT RCWA convention -- ridge centre
``0.5 + shear * (zeta - 0.5)`` in period fractions, half-open ridge
``[centre - duty/2, centre + duty/2)``, ``zeta`` sampled per slice by the same
``rule`` as the duty -- so the two packages build the IDENTICAL staircase, PMM
laterally exact and RCWA lattice-quantised.

MEASURED (this file's pins, on this box)
----------------------------------------
* ``shear=0`` is BIT-identical to the pre-``shear`` builder: 0 differing
  segment widths over 9 ``(duty_b, duty_t, n_slices, rule)`` cases x both
  builders x {kwarg omitted, ``shear=0.0``}.  Pure geometry -> exact assert.
* Staircase-GEOMETRY identity with RCWA: rasterizing the PMM segments on the
  RCWA pixel-centre lattice reproduces its ``eps_cell`` with 0 mismatched
  pixels over ``shear`` in ``[-1.7, 2.5]`` x 3 duty configs x
  ``n_x`` in {64, 256, 1024, 4096}, wrapping cases included.
* Cross-package PHYSICS: at fixed ``n_slices`` the RCWA answer converges to the
  PMM one as ``n_x`` grows.  Measured on the TE row (``E_y`` is tangential to
  every wall, so RCWA's Fourier truncation is ~1e-9-clean there and the
  residual gap IS the raster quantisation), ``eps_ridge = 2.1``, ``nox = 31``,
  ``n_slices = 6``, ``shear = 0.35``::

      n_x   |  gap      | vs n_x=512
       512  | 4.683e-04 |
      2048  | 1.023e-04 |  4.6x
      8192  | 9.362e-06 | 50.0x

  The FULL-row (TE+TM) gap instead SATURATES at RCWA's Fourier floor
  (measured 4.7e-04 at ``nox = 31/61``, 1.8e-04 at eps 2.1) -- expected, and
  the reason the convergence pin reads the TE row.  ``shear`` only PHASE-shifts
  the cell's Fourier coefficients, so that floor is shear-invariant (measured
  identical 4.6888e-04 at ``shear`` 0.35 and 1.4).

TASK B -- the measured staircase error law (documentation defect FOUND)
======================================================================

``add_tapered_grating``'s cost note claimed the staircase "converges as
``O(1/n_slices^2)``".  Measured (RCWA ``raster='area'``, so the realised duty
is exact and the ONLY z error is the staircase; reference ``n_slices = 768``),
the max-over-orders error DOES approach ``O(1/n^2)`` asymptotically, but the
per-channel exponents SPLIT: the wall-tangential (TE / incident ``E_y``)
channels reach ``p ~ 3``, while the wall-NORMAL (TM / incident ``E_x``)
transmissions crawl at ``p ~ 0.85-1`` -- the staircase re-entrant-corner
singularity.  The pins below lock the measured law so a future "faster
staircase" claim has to beat a number.
"""
import os
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.rcwa import RCWAStack

_P = 1.0e-6
_WL = 0.633e-6
_TH = 0.30e-6
_ER, _EG = 4.0 + 0j, 1.0 + 0j
_ANG = 0.17

# (duty_bottom, duty_top, n_slices, rule) -- the shear=0 regression matrix
_CASES = [(0.5, 0.5, 5, "midpoint"), (0.6, 0.3, 6, "midpoint"),
          (0.6, 0.3, 7, "trapezoid"), (0.35, 0.85, 9, "midpoint"),
          (0.9, 0.1, 12, "trapezoid"), (0.62, 0.30, 6, "midpoint"),
          (0.5, 0.5, 1, "midpoint")]


def _grating(db, dt, ns, rule, shear=None, degree=8, eps_ridge=_ER,
             thickness=_TH):
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=degree,
                  far_field_orders=11)
    kw = {} if shear is None else dict(shear=shear)
    st.add_tapered_grating(thickness, eps_ridge=eps_ridge, eps_groove=_EG,
                           duty_bottom=db, duty_top=dt, n_slices=ns,
                           rule=rule, **kw)
    return st


def _widths(st):
    return [[w for w, _e in segs] for _t, segs, _s in st._layers]


def _epsxx(st):
    return [[complex(np.asarray(e)[0, 0]) for _w, e in segs]
            for _t, segs, _s in st._layers]


# ===================================================================== TASK A
# ---- REGRESSION locks (these PASS on a pre-change worktree) ----------------

def test_regression_centred_slice_layout_is_the_historical_triple():
    """REGRESSION (passes pre-change).  With the ``shear`` kwarg OMITTED every
    patterned slice is still EXACTLY ``[0.5*(1-duty), duty, 0.5*(1-duty)]``
    with the duty sampled by the documented rule -- bit-for-bit.

    Pure wall arithmetic with no eigensolve downstream, so an exact assert is
    the right instrument (TOLERANCE_POLICY): the post-change builder reaches
    the same three floats only because halving is exact in binary floating
    point (``(1-d)/2 == 0.5 - d/2``), and that identity is what this locks."""
    for db, dt, ns, rule in _CASES:
        got = _widths(_grating(db, dt, ns, rule))
        assert len(got) == ns
        for k, ws in enumerate(got):
            zeta = ((k + 0.5) / ns if rule == "midpoint"
                    else 0.5 * (k / ns + (k + 1) / ns))
            duty = dt + (db - dt) * zeta
            edge = 0.5 * (1.0 - duty)
            assert ws == [edge, duty, edge], (db, dt, ns, rule, k, ws)


def test_regression_vertical_limit_still_matches_a_single_layer():
    """REGRESSION (passes pre-change).  ``duty_top == duty_bottom`` with the
    kwarg omitted is a vertical binary grating -> the staircase of identical
    slices reproduces one hand-built layer."""
    o_t, R_t, T_t, _j = _grating(0.5, 0.5, 5, "midpoint",
                                 degree=10).set_source(_WL,
                                                       angle=_ANG).solve()
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=10,
                  far_field_orders=11)
    st.add_layer(_TH, segments=[(0.25, _EG), (0.5, _ER), (0.25, _EG)])
    o_v, R_v, T_v, _ = st.set_source(_WL, angle=_ANG).solve()
    assert np.array_equal(o_t, o_v)            # integer order labels
    assert np.max(np.abs(R_t - R_v)) < 1e-10   # measured 0.0 (same geometry)
    assert np.max(np.abs(T_t - T_v)) < 1e-10


# ---- NEW capability (these FAIL on a pre-change worktree: TypeError) -------

def test_shear_zero_is_bit_identical_to_the_omitted_kwarg():
    """``shear=0.0`` must not perturb ANY existing stack: identical segment
    widths AND permittivities on both builders and both rules."""
    for db, dt, ns, rule in _CASES:
        a, b = _grating(db, dt, ns, rule), _grating(db, dt, ns, rule,
                                                    shear=0.0)
        assert _widths(a) == _widths(b), (db, dt, ns, rule)
        assert _epsxx(a) == _epsxx(b)
    for shear_kw in ({}, dict(shear=0.0)):
        st = PMMStack(_P, n_substrate=1.5, degree=8, far_field_orders=11)
        st.add_tapered_ridges(_TH, ridges=[(0.5 * _P, 0.30 * _P, 0.62 * _P,
                                            _ER)],
                              eps_groove=_EG, n_slices=6, **shear_kw)
        if not shear_kw:
            base = _widths(st)
        else:
            assert _widths(st) == base


def test_shear_zero_solves_identically_same_process():
    """Same-process determinism: bit-identical geometry -> bit-identical solve
    (measured max|diff| = 0.0; pinned at 1e-12 so a BLAS reduction reorder on
    a CI runner cannot fail the intent)."""
    a = _grating(0.62, 0.30, 6, "midpoint", degree=10)
    b = _grating(0.62, 0.30, 6, "midpoint", shear=0.0, degree=10)
    oa, Ra, Ta, Ja = a.set_source(_WL, angle=_ANG).solve()
    ob, Rb, Tb, Jb = b.set_source(_WL, angle=_ANG).solve()
    assert np.array_equal(oa, ob)
    for X, Y in ((Ra, Rb), (Ta, Tb), (Ja, Jb)):
        assert np.max(np.abs(np.asarray(X) - np.asarray(Y))) < 1e-12


def test_shear_geometry_is_the_rcwa_staircase_exactly():
    """CROSS-PACKAGE GEOMETRY IDENTITY.  Rasterizing the PMM segments on the
    RCWA pixel-centre lattice must reproduce RCWA's own ``eps_cell`` column
    for every slice -- pure mask arithmetic (no eigensolve), so exact.

    This is the convention pin: it fails for ANY other centre law, sign, or
    zeta sampling, and it exercises the wrap layout (|shear| > 1 walks the
    ridge clean across the cell).

    The RCWA side is the DOCUMENTED rule spelled out inline -- the half-open,
    wrap-aware membership test ``(u - lo) mod 1 < duty`` with
    ``lo = centre - duty/2`` and ``centre = 0.5 + shear*(zeta - 0.5)``,
    ``zeta = (k + 0.5)/n_slices`` (``RCWAStack.add_tapered_grating``'s
    ``_profile`` at ``raster='hard'``) -- rather than an import of that
    module's private rasterizer, so the pin states the contract instead of
    inheriting it."""
    for shear in (-1.7, -0.9, -0.35, 0.0, 0.2, 0.35, 0.75, 1.0, 1.7, 2.5):
        for db, dt, ns in ((0.62, 0.30, 6), (0.5, 0.5, 4), (0.85, 0.35, 9)):
            layers = _grating(db, dt, ns, "midpoint", shear=shear)._layers
            for n_x in (64, 256, 1024, 4096):
                u = (np.arange(n_x) + 0.5) / n_x
                for k, (_t, segs, _s) in enumerate(layers):
                    edges = np.concatenate(
                        [[0.0], np.cumsum([w for w, _e in segs])])
                    edges[-1] = 1.0
                    pmm = np.empty(n_x, dtype=complex)
                    for j, (_w, e) in enumerate(segs):
                        pmm[(u >= edges[j]) & (u < edges[j + 1])] = \
                            complex(np.asarray(e)[0, 0])
                    zeta = (k + 0.5) / ns
                    duty = dt + (db - dt) * zeta
                    lo = 0.5 + shear * (zeta - 0.5) - 0.5 * duty
                    cov = np.mod(u - lo, 1.0) < duty      # half-open, wrapping
                    rcwa = np.where(cov, _ER, _EG)
                    assert np.array_equal(pmm, rcwa), (shear, db, dt, ns,
                                                       n_x, k)


def test_ridge_slice_segments_invariants():
    """The wrap-aware slice builder: every width > 0, the widths sum to 1 to
    within an ULP, and the realised ridge fraction is the requested duty
    EXACTLY (the PMM lateral-exactness contract -- RCWA can only quantise it).

    Pure geometry: measured over 4000 random ``(duty, centre)`` plus the
    boundary-coincidence family (a wall landing exactly on ``u = 0`` / ``1``),
    max|sum - 1| = 1.11e-16 and max|ridge_fraction - duty| = 0.0."""
    rng = np.random.default_rng(20260727)
    cases = [(float(d), float(c)) for d, c in
             zip(rng.uniform(1e-6, 1 - 1e-6, 4000),
                 rng.uniform(-3.0, 3.0, 4000))]
    cases += [(0.5, 0.25), (0.5, 0.75), (0.5, 1.25), (0.25, 0.125),
              (0.5, 0.5), (0.3, 0.15), (0.3, 1.0), (0.3, 0.0), (0.5, -0.25),
              (0.999999, 0.5), (1e-9, 0.5), (0.5, 0.75 + 1e-16),
              (0.5, 0.25 - 1e-16), (0.4, 0.2 + 5e-17)]
    for duty, centre in cases:
        segs = PMMStack._ridge_slice_segments(duty, centre, _ER, _EG)
        ws = np.array([w for w, _e in segs])
        assert ws.size in (2, 3), (duty, centre, segs)
        assert np.all(ws > 0.0), (duty, centre, segs)
        assert abs(float(ws.sum()) - 1.0) < 1e-14, (duty, centre, ws)
        ridge = float(sum(w for w, e in segs if e == _ER))
        assert abs(ridge - duty) < 5e-16, (duty, centre, ridge)


def test_tapered_ridges_shear_reproduces_the_grating():
    """A single ridge centred at ``period/2`` with the same ``shear`` must
    rebuild ``add_tapered_grating``'s staircase (measured max width difference
    1.2e-16 -- the two constructions differ only in rounding order)."""
    for shear in (-1.3, -0.4, 0.0, 0.4, 0.9, 1.6):
        for db, dt, ns in ((0.62, 0.30, 6), (0.45, 0.45, 5)):
            A = _grating(db, dt, ns, "midpoint", shear=shear)
            B = PMMStack(_P, n_substrate=1.5, degree=8, far_field_orders=11)
            B.add_tapered_ridges(_TH, ridges=[(0.5 * _P, dt * _P, db * _P,
                                               _ER)],
                                 eps_groove=_EG, n_slices=ns, shear=shear)
            wa, wb = _widths(A), _widths(B)
            assert [len(r) for r in wa] == [len(r) for r in wb], (shear, db)
            assert _epsxx(A) == _epsxx(B)
            for ra, rb in zip(wa, wb):
                assert np.max(np.abs(np.array(ra) - np.array(rb))) < 1e-14


def test_shear_is_recorded_in_the_taper_recipe():
    """``_resliced_clone`` (and therefore ``stabilize='slices'``) REPLAYS the
    recorded builder, so a new kwarg that is not recorded would silently
    re-slice an UNSHEARED stack and hand back a bogus consensus."""
    for shear in (0.35, 0.9):
        st = _grating(0.62, 0.30, 5, "midpoint", shear=shear)
        (_i0, cnt, method, kw), = st._taper_recipes
        assert method == "add_tapered_grating" and cnt == 5
        assert kw["shear"] == shear
        clone = st._resliced_clone(+1)
        direct = _grating(0.62, 0.30, 6, "midpoint", shear=shear)
        assert _widths(clone) == _widths(direct)
        st2 = PMMStack(_P, n_substrate=1.5, degree=8, far_field_orders=11)
        st2.add_tapered_ridges(_TH, ridges=[(0.35 * _P, 0.2 * _P, 0.3 * _P,
                                             _ER)],
                               eps_groove=_EG, n_slices=4, shear=shear)
        assert st2._taper_recipes[0][3]["shear"] == shear
        assert len(st2._resliced_clone(-1)._layers) == 3


def test_solve_vs_wavelength_carries_the_shear():
    """The assemble-once sweep must see the SHEARED geometry: it rebuilds only
    the (dispersive) permittivities, so a shear lost between the builder and
    the sweep would silently solve the centred structure (measured max
    |sweep - per-wavelength| = 0.0)."""
    wls = np.linspace(0.55e-6, 0.70e-6, 3)

    def _eps(w):
        return 4.0 + 0.5 * (w / 0.633e-6 - 1.0)

    for shear in (0.35, 0.9):
        st = PMMStack(_P, n_substrate=1.5, degree=10, far_field_orders=11)
        st.add_tapered_grating(_TH, eps_ridge=_eps, eps_groove=_EG,
                               duty_bottom=0.62, duty_top=0.30, n_slices=4,
                               shear=shear)
        o_sw, R_sw, T_sw = st.solve_vs_wavelength(wls, angle=_ANG)
        assert R_sw.shape == (len(wls), 2, len(o_sw))
        for iw, w in enumerate(wls):
            one = _grating(0.62, 0.30, 4, "midpoint", shear=shear, degree=10,
                           eps_ridge=_eps(float(w)))
            o1, R1, T1, _j = one.set_source(float(w), angle=_ANG).solve()
            for m in o1:
                if m in o_sw:
                    js = int(np.where(o_sw == m)[0][0])
                    j1 = int(np.where(o1 == m)[0][0])
                    assert np.allclose(R_sw[iw, :, js], R1[:, j1],
                                       rtol=0, atol=1e-10)
                    assert np.allclose(T_sw[iw, :, js], T1[:, j1],
                                       rtol=0, atol=1e-10)


def test_shear_rejects_non_finite():
    for bad in (float("nan"), float("inf"), float("-inf")):
        st = PMMStack(_P, degree=8, far_field_orders=11)
        with pytest.raises(ValueError, match="shear must be finite"):
            st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                                   duty_bottom=0.5, n_slices=3, shear=bad)
        with pytest.raises(ValueError, match="shear must be finite"):
            st.add_tapered_ridges(_TH, ridges=[(0.5 * _P, 0.2 * _P,
                                                0.3 * _P, _ER)],
                                  eps_groove=_EG, n_slices=3, shear=bad)


def test_tapered_ridges_shear_keeps_the_overlap_guard():
    """Sheared ridges still collide, and the guard must fire on the SHEARED
    geometry rather than on the unsheared centres."""
    st = PMMStack(_P, degree=8, far_field_orders=11)
    with pytest.raises(ValueError, match="ridges overlap"):
        st.add_tapered_ridges(_TH, ridges=[(0.3 * _P, 0.4 * _P, 0.4 * _P, _ER),
                                           (0.6 * _P, 0.4 * _P, 0.4 * _P,
                                            _ER)],
                              eps_groove=_EG, n_slices=3, shear=0.5)


# ---- the cross-package capability oracle -----------------------------------

def _pmm_te(shear, ns, eps_ridge, degree):
    o, R, T, _j = _grating(0.62, 0.30, ns, "midpoint", shear=shear,
                           degree=degree,
                           eps_ridge=eps_ridge).set_source(
        _WL, angle=_ANG).solve()
    return np.asarray(o), np.asarray(R)[1], np.asarray(T)[1]


def _rcwa_te(shear, ns, eps_ridge, nox, n_x):
    st = RCWAStack(_P, n_substrate=1.5, n_superstrate=1.0, n_orders=nox)
    st.add_tapered_grating(_TH, eps_ridge=eps_ridge, eps_groove=_EG,
                           duty_bottom=0.62, duty_top=0.30, n_slices=ns,
                           n_x=n_x, shear=shear)
    o, R, T = st.set_source(_WL, theta=_ANG).solve().efficiencies()
    o = np.asarray(o)
    return (o[:, 0] if o.ndim == 2 else o), np.asarray(R)[1], np.asarray(T)[1]


def _gap(a, b):
    oa, Ra, Ta = a
    ob, Rb, Tb = b
    d = 0.0
    for m in (-2, -1, 0, 1, 2):
        ia, ib = np.where(oa == m)[0], np.where(ob == m)[0]
        if ia.size and ib.size:
            d = max(d, abs(float(Ra[ia[0]] - Rb[ib[0]])),
                    abs(float(Ta[ia[0]] - Tb[ib[0]])))
    return d


@pytest.mark.parametrize("shear", [0.35, 0.90])
def test_cross_package_staircase_converges_in_n_x(shear):
    """CAPABILITY ORACLE.  At the SAME ``(n_slices, duties, shear)`` the RCWA
    staircase is the PMM staircase QUANTISED to its ``n_x`` lattice, so the gap
    must SHRINK as ``n_x`` grows -- it is not an independent-implementation
    tolerance but a convergence statement.

    Read on the TE row only: ``E_y`` is tangential to every wall, so RCWA's
    Fourier truncation there is ~1e-9 (measured) and the gap IS the raster
    error; the TM row saturates on RCWA's ``O(1/nox)`` Fourier floor
    (measured 4.7e-04 at nox 31 AND 61 -- refining ``n_x`` cannot move it).
    ``shear = 0.90`` walks the ridge across ``u = 1`` (the WRAP layout).

    MEASURED (eps_ridge 2.1, nox 31, n_slices 6, PMM degree 12)::

        shear | n_x=512    2048       8192     | 512/8192
        0.35  | 4.683e-04  1.023e-04  9.362e-06 |  50.0x
        0.90  | 1.073e-04  3.302e-05  1.099e-05 |   9.8x
    """
    er = 2.1 + 0j
    ref = _pmm_te(shear, 6, er, 12)
    tot = [float(np.asarray(x).sum()) for x in ref[1:]]
    assert abs(sum(tot) - 1.0) < 1e-7           # TE closure (measured ~1e-11)
    g = {n_x: _gap(_rcwa_te(shear, 6, er, 31, n_x), ref)
         for n_x in (512, 2048, 8192)}
    assert g[512] > 2e-5, g                     # the trend must be resolvable
    assert g[2048] < g[512], g                  # monotone in n_x
    assert g[8192] < g[512] / 3.0, g            # measured 50x / 9.8x
    assert g[8192] < 2e-4, g                    # measured 9.4e-6 / 1.1e-5


def test_sheared_grating_zero_shear_is_a_plain_vertical_layer():
    """``shear = 0`` must degenerate to ONE VERTICAL layer with the historical
    centred triple (pure geometry -> exact), and solve identically to a
    hand-built one."""
    st = PMMStack(_P, n_substrate=1.5, degree=10, far_field_orders=11)
    st.add_sheared_grating(_TH, eps_ridge=_ER, eps_groove=_EG, duty=0.45,
                           shear=0.0)
    (thk, segs, slant), = st._layers
    assert thk == _TH and slant == 0.0
    assert [w for w, _e in segs] == [0.275, 0.45, 0.275]
    hand = PMMStack(_P, n_substrate=1.5, degree=10, far_field_orders=11)
    hand.add_layer(_TH, segments=[(0.275, _EG), (0.45, _ER), (0.275, _EG)])
    a = st.set_source(_WL, angle=_ANG).solve()
    b = hand.set_source(_WL, angle=_ANG).solve()
    assert np.array_equal(a[0], b[0])
    for x, y in zip(a[1:], b[1:]):
        assert np.max(np.abs(np.asarray(x) - np.asarray(y))) < 1e-12


def test_sheared_grating_geometry_is_the_staircase_zeta_limit():
    """The exact-slant layer's u-frame segments are the ``zeta -> 0`` slice of
    the staircase of the SAME ``shear`` (the slant frame ``u = x - z tan(phi)``
    is anchored at the layer TOP), and ``slant_angle`` is
    ``+arctan(shear * period / thickness)`` -- the SIGN is the load-bearing
    half: the opposite sign solves a completely different structure (MEASURED
    0.32 away from the staircase limit, against 2.9e-03 for the right one)."""
    for shear, duty in ((0.35, 0.45), (-0.35, 0.45), (0.9, 0.55),
                        (1.4, 0.45)):
        st = PMMStack(_P, n_substrate=1.5, degree=8, far_field_orders=11)
        st.add_sheared_grating(_TH, eps_ridge=_ER, eps_groove=_EG, duty=duty,
                               shear=shear)
        (_t, segs, slant), = st._layers
        assert abs(slant - np.arctan(shear * _P / _TH)) < 1e-15
        assert slant * shear > 0.0                     # the sign convention
        fine = PMMStack(_P, n_substrate=1.5, degree=8, far_field_orders=11)
        fine.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                                 duty_bottom=duty, duty_top=duty,
                                 n_slices=2000, shear=shear)
        w_slant = np.array([w for w, _e in segs])
        w_top = np.array([w for w, _e in fine._layers[0][1]])
        assert w_slant.shape == w_top.shape, (shear, duty)
        # the staircase samples zeta = 1/4000, so the walls differ by
        # shear/4000 (measured max 1.8e-04 at shear 0.35 -- pinned loosely)
        assert np.max(np.abs(w_slant - w_top)) < 2.0 * abs(shear) / 2000.0


def test_sheared_grating_matches_the_staircase_within_its_measured_floor():
    """CAPABILITY ORACLE for the exact-slant route: it must land on the
    staircase's answer to the slant path's own per-order floor, at a small
    fraction of the cost -- and be BETTER than the staircase at a matched
    budget.

    MEASURED (``P = 1 um``, ``wl = 633 nm``, ``d = 300 nm``, ``eps_ridge = 4``,
    ``duty = 0.45``, ``shear = 0.35`` = a 49.4 deg wall, ``theta = 0.17``,
    ``degree = 10`` staircase / 12 slant; error vs the ``n_slices = 20``
    staircase)::

        staircase ns=4  2.59e-02  0.25 s     slant deg=8   3.16e-03  0.03 s
        staircase ns=8  6.93e-03  2.27 s     slant deg=12  2.93e-03  0.05 s
        staircase ns=12 2.63e-03  7.53 s     slant deg=16  2.91e-03  0.13 s

    The slant PLATEAUS at ~2.9e-03 (the documented oblique+slant wall-normal
    per-order floor) -- pinned as a BAND, not a limit: it must beat the
    ``n_slices = 4`` staircase and land inside 1e-02 of the ``n_slices = 10``
    one, which it does with ~3x headroom on both sides."""
    duty, shear = 0.45, 0.35

    def _stair(ns):
        st = PMMStack(_P, n_substrate=1.5, degree=10, far_field_orders=11)
        st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                               duty_bottom=duty, duty_top=duty, n_slices=ns,
                               shear=shear)
        o, R, T, _j = st.set_source(_WL, angle=_ANG).solve()
        return np.asarray(o), np.asarray(R), np.asarray(T)

    def _slant(deg):
        st = PMMStack(_P, n_substrate=1.5, degree=deg, far_field_orders=11)
        st.add_sheared_grating(_TH, eps_ridge=_ER, eps_groove=_EG, duty=duty,
                               shear=shear)
        o, R, T, _j = st.set_source(_WL, angle=_ANG).solve()
        return np.asarray(o), np.asarray(R), np.asarray(T)

    def _g(a, b):
        d = 0.0
        for m in (-2, -1, 0, 1, 2):
            ia, ib = np.where(a[0] == m)[0], np.where(b[0] == m)[0]
            if ia.size and ib.size:
                d = max(d,
                        float(np.max(np.abs(a[1][:, ia[0]] - b[1][:, ib[0]]))),
                        float(np.max(np.abs(a[2][:, ia[0]] - b[2][:, ib[0]]))))
        return d

    ref = _stair(10)
    s4 = _stair(4)
    sl = _slant(12)
    assert _g(s4, ref) > 1e-2                  # measured 2.3e-02 vs ns=10
    assert _g(sl, ref) < 1e-2                  # measured 2.6e-03
    assert _g(sl, ref) < _g(s4, ref) / 3.0     # measured ~9x better
    for pol in (0, 1):                         # the slant path's own closure
        assert abs(float(sl[1][pol].sum() + sl[2][pol].sum()) - 1.0) < 1e-3


def test_sheared_grating_validation_and_restrictions():
    for bad in (0.0, 1.0, -0.1, 1.2, float("nan")):
        st = PMMStack(_P, degree=8, far_field_orders=11)
        with pytest.raises(ValueError, match="duty must be strictly inside"):
            st.add_sheared_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                                   duty=bad, shear=0.2)
    for bad in (float("nan"), float("inf")):
        st = PMMStack(_P, degree=8, far_field_orders=11)
        with pytest.raises(ValueError, match="shear must be finite"):
            st.add_sheared_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                                   duty=0.5, shear=bad)
    st = PMMStack(_P, n_substrate=1.5, degree=8, far_field_orders=11)
    st.add_sheared_grating(_TH, eps_ridge=_ER, eps_groove=_EG, duty=0.45,
                           shear=0.35)
    with pytest.raises(NotImplementedError):        # slanted -> sweep rejects
        st.solve_vs_wavelength([_WL], angle=_ANG)
    with pytest.raises(NotImplementedError):        # ... and internal fields
        st.set_source(_WL, angle=_ANG).solve(retain_internal=True)
    with pytest.raises(NotImplementedError):        # ... and conical
        st.set_source(_WL, angle=_ANG, phi=0.3).solve()
    keyed = PMMStack(_P, n_substrate=1.5, degree=8, far_field_orders=11)
    keyed.add_sheared_grating(_TH, eps_ridge="Si", eps_groove=_EG, duty=0.45,
                              shear=0.35)
    with pytest.raises(NotImplementedError):        # ... and the key sweep
        keyed.prepare()
    # stabilize='slices' has no recipe to re-slice -> since the 2026-07-28
    # union-grid audit it falls back to the min_feature-perturbation consensus
    # instead of announcing that it skipped.  A sheared grating is ONE exact
    # slanted layer with NO z-staircase, so there are no colliding cross-layer
    # walls and the consensus is a no-op: it must run quietly and not raise.
    st2 = PMMStack(_P, n_substrate=1.5, degree=8, far_field_orders=11)
    st2.add_sheared_grating(_TH, eps_ridge=_ER, eps_groove=_EG, duty=0.45,
                            shear=0.35)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        st2.set_source(_WL, angle=_ANG).solve(stabilize="slices")
    msgs = [str(w.message) for w in wlist]
    assert not any("no taper builder recorded" in m for m in msgs)
    assert not any("min_feature` was perturbed" in m for m in msgs)


def test_staircase_error_law_is_second_order():
    """TASK B lock: the documented ``O(1/n_slices^2)`` staircase law, measured
    on the cross-package oracle that ISOLATES it -- the RCWA twin at
    ``raster='area'`` (its realised duty is then exact, so the only z error is
    the staircase; the DEFAULT ``'hard'`` raster injects ``O(1/n_x)`` duty
    quantisation that masks the law entirely).

    MEASURED with ``nox = 15``, ``n_x = 1024``, reference ``n_slices = 256``:
    the error falls by a factor ~3-4 per doubling of ``n_slices`` from 4 to 32
    (measured 6.355e-03 / 2.153e-03 / 5.640e-04 / 1.399e-04, ratios
    2.95 / 3.82 / 4.03) -- pinned as ``> 2.5`` per doubling.

    CALIBRATION NOTE (CI red at d3941f5, py3.12): the original window
    (ns 16..128, reference 512) put the tail error at 8.4e-6 locally, INSIDE
    the catalogued CI multi-slice drift band (~1e-5 -- unpinned BLAS threads on
    the fast gate drift a many-layer cascade run-to-run; the 64-slice stack
    class measured 2.2e-5 on CI vs 2e-7 locally in W8).  CI 3.12 read the
    64->128 ratio at 2.18 there while the SAME code was green at a210de4 --
    coin-flip noise, not physics: the a210de4-vs-d3941f5 oracle solves are
    bit-identical for every n_slices (measured in clean worktrees).  The
    window is shifted up the error curve so its smallest error (1.4e-4) sits
    14x above that noise floor."""
    def _obs(ns):
        st = RCWAStack(_P, n_substrate=1.5, n_superstrate=1.0, n_orders=15)
        st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                               duty_bottom=0.62, duty_top=0.30, n_slices=ns,
                               n_x=1024, raster="area")
        o, R, T = st.set_source(_WL, theta=_ANG).solve().efficiencies()
        o = np.asarray(o)
        o = o[:, 0] if o.ndim == 2 else o
        idx = [int(np.where(o == m)[0][0]) for m in (-1, 0, 1)]
        return np.concatenate([np.asarray(R)[:, idx].ravel(),
                               np.asarray(T)[:, idx].ravel()])

    ref = _obs(256)
    err = {n: float(np.max(np.abs(_obs(n) - ref))) for n in (4, 8, 16, 32)}
    assert err[4] > 1e-3, err                  # measured 6.36e-03
    for n in (8, 16, 32):
        assert err[n // 2] / err[n] > 2.5, err  # measured 2.95 / 3.82 / 4.03


def test_pmm_lateral_exactness_beats_the_rcwa_raster():
    """The POINT of the PMM twin: at the same staircase the PMM answer needs no
    ``n_x`` at all.  Its own convergence knob (``degree``) reaches the TE
    reference orders of magnitude below the best RCWA raster here (measured
    |deg10 - deg20| = 7.5e-10 vs an RCWA raster gap of 9.4e-06 at
    ``n_x = 8192``)."""
    er = 2.1 + 0j
    ref = _pmm_te(0.35, 6, er, 20)
    assert _gap(_pmm_te(0.35, 6, er, 10), ref) < 1e-7      # measured 7.5e-10
    assert _gap(_rcwa_te(0.35, 6, er, 31, 8192), ref) > 1e-6   # 9.4e-06
