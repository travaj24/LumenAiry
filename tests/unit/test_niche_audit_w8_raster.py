"""Wave-8 pins -- the ``eps_cell`` RASTERIZATION CONTRACT
(``lumenairy/elements/rcwa/stack.py`` tapered generators + the pixel semantics
of every ``eps_cell``-accepting entry point).

The W7 rcwa audit deferred this deliberately: the tapered / sheared generators
rasterized their analytic shapes with a SYMMETRIC ``|dist| < half``, which
excludes BOTH walls, so a wall landing EXACTLY on a pixel centre lost one
pixel -- the W7-A defect class (a duty quantised to a grid) but in the
USER-FACING pixel cell, where changing the semantics is a CONTRACT decision.
The owner's decision, and this file's three parts:

W8-1  BOUNDARY COINCIDENCE (FOUND + FIXED).  Half-open ``[lo, lo + w)`` --
      lower wall inclusive, upper exclusive -- matching the analytic 1-D paths
      (``oned.py``: "the ridge occupies ``[0, duty)``"),
      ``PMMStack._ridges_to_segments`` (``lo <= mid < hi``) and
      ``SegmentStackGeometry.to_rcwa_stack`` (``(xs >= lo) & (xs < hi)``).
      Measured pre-fix at ``shear=0.5, duty=0.5, n_slices=128, n_x=256``: ALL
      128 slices realised duty ``127/256 = 0.49609375`` (3.906e-03 short) --
      the recorded W7 outlier, reproduced exactly.  ``n_x == 2 n_slices`` is a
      whole coincidence FAMILY; on a clean-closure case (``eps_ridge=4``,
      ``M=7``, 64 slices, coincidence at ``n_x=128``) it moved the zeroth
      orders by 1.802e-02, now 1.552e-04 (the ordinary ``O(1/n_x)``).

W8-2  PIXEL SEMANTICS (documented; measured here).  ``eps_cell[j, i]`` is a
      NODE point sample at ``(j Px/Sx, i Py/Sy)`` -- what
      ``fft2(cell)/(Sx Sy)`` interpolates through.  The generators sample the
      pixel-CENTRE lattice, half a pixel away; for a band-limited cell that is
      exactly a rigid ``-P/(2 Sx)`` translation (efficiency-invariant), and for
      a hard raster it is a second ``O(1/Sx)`` quantisation that shrinks with
      the first.

W8-3  OPT-IN AREA WEIGHTING (``raster='hard'|'area'``, default ``'hard'``,
      bit-preserved).  The measured verdict, per polarization AND
      formulation: ``'area'`` is 1-3 orders of magnitude better with the
      DEFAULT ``'laurent'`` (both polarizations), and WORSE (up to 10.6x) for
      the wall-NORMAL polarization under ``'li'`` -- the inverse rule assumes a
      sharp interface and wants the HARMONIC, not the arithmetic, boundary
      average.  Shipped opt-in with that recorded.

MEASURED PRE-FIX CHECK: 22 pins here, of which 16 FAIL on a clean ``e37d7b7``
worktree.  The 6 that pass are the documented NON-DISCRIMINATORS -- they encode
the CONTRACT the fix is built on, not the fix itself:

  test_w8_non_coincident_shears_were_already_exact_and_stay_exact  (the control
      arm: this is WHY the defect read as a convergence outlier)
  test_w8_bit_identical_off_coincidence     (pure-math equivalence of the old
      and new predicates away from a coincidence)
  test_w8_pixel_is_a_node_sample_of_the_factorization
  test_w8_generator_lattice_is_the_pixel_midpoint
  test_w8_rigid_translation_leaves_efficiencies_invariant
  test_w8_pmm_tapered_siblings_are_exempt   (the PMM exemption, by measurement)

Solves are deliberately small (``n_orders <= 9``); no exact float pins on
solver output (eigensolves drift ~1e-11 cross-platform), only measured
tolerances.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.rcwa import RCWAStack, rcwa_efficiency_1d
from lumenairy.elements.rcwa.twod import (
    _eps_convolution_2d,
    _harmonic_orders_2d,
)

try:                                   # absent on a pre-v5.31 (pre-W8) tree
    from lumenairy.elements.rcwa.stack import _raster_cover_1d
except ImportError:                    # pragma: no cover -- pre-fix tree only
    _raster_cover_1d = None            # so the MODULE still collects there and
    #                                    each pin fails on its own merits

_C = np.complex128

# shared measurement geometry (matches the docstring tables)
_WL, _P, _D, _M = 0.633e-6, 1.0e-6, 0.30e-6, 9
_N_RIDGE, _N_GROOVE = 2.0, 1.0
_ER, _EG = _N_RIDGE ** 2, _N_GROOVE ** 2
_DUTY = 0.37
_ROW = {"TE": 1, "TM": 0}          # solve rows: 0 = incident Ex (p), 1 = Ey (s)


# ===========================================================================
# shared oracles / helpers
# ===========================================================================

def _old_symmetric_mask(u, centre, half):
    """The PRE-v5.31 rasterization predicate, verbatim from the three tapered
    builders: ``|wrap(u - centre)| < half``.  Excludes BOTH walls."""
    dist = np.abs((u - centre + 0.5) % 1.0 - 0.5)
    return dist < half


def _half_open_mask(u, lo, width):
    """The v5.31 convention, written INDEPENDENTLY of the library helper: the
    feature owns ``[lo, lo + width)``, wrap-aware."""
    if width >= 1.0:
        return np.ones(np.shape(u), dtype=bool)
    if width <= 0.0:
        return np.zeros(np.shape(u), dtype=bool)
    return np.mod(u - lo, 1.0) < width


def _cover(u, lo, width, raster):
    """The library helper, with an explicit failure on a pre-W8 tree."""
    assert _raster_cover_1d is not None, (
        "pre-v5.31 tree: lumenairy.elements.rcwa.stack._raster_cover_1d does "
        "not exist (the W8 rasterization contract is not implemented)")
    return _raster_cover_1d(u, lo, width, raster)


def _realised_duties(*, n_slices, n_x, shear, duty=0.5):
    """Per-slice realised duty of a tapered grating (DEFAULT raster), read
    straight out of the cell: with ridge 4 / groove 1 the cell MEAN is
    ``1 + 3*duty``, and the mean IS the DC Fourier coefficient the solver
    sees."""
    st = RCWAStack(_P, n_orders=3)
    st.add_tapered_grating(0.3e-6, eps_ridge=4.0, eps_groove=1.0,
                           duty_bottom=duty, duty_top=duty,
                           n_slices=n_slices, n_x=n_x, shear=shear)
    return np.array([(float(np.real(np.asarray(L.data)[:, 0].mean())) - 1.0)
                     / 3.0 for L in st._layers])


def _cell_of(*, n_x, raster, n_slices=1, shear=0.0, duty=_DUTY):
    st = RCWAStack(_P, n_orders=_M)
    st.add_tapered_grating(_D, eps_ridge=_ER, eps_groove=_EG,
                           duty_bottom=duty, duty_top=duty,
                           n_slices=n_slices, n_x=n_x, shear=shear,
                           raster=raster)
    return [np.asarray(L.data) for L in st._layers]


def _solve_cells(cells, *, formulation="laurent", theta=0.0, n_orders=_M):
    st = RCWAStack(_P, n_orders=n_orders)
    for c in cells:
        st.add_layer(_D / len(cells), eps_cell=c, formulation=formulation)
    res = st.set_source(_WL, theta=theta).solve()
    o, R, T = res.efficiencies()
    o = np.asarray(o)
    i0 = int(np.where(o == 0)[0][0])
    i1 = int(np.where(o == 1)[0][0])
    return np.asarray(R), np.asarray(T), i0, i1


def _quad(cells, pol, *, formulation="laurent", theta=0.0):
    """``(R0, T0, R+1, T+1)`` for one polarization."""
    R, T, i0, i1 = _solve_cells(cells, formulation=formulation, theta=theta)
    r = _ROW[pol]
    return np.array([R[r, i0], T[r, i0], R[r, i1], T[r, i1]])


def _quad_exact(pol, formulation, theta=0.0):
    """The EXACT analytic 1-D oracle for the vertical duty-0.37 grating (the
    post-W7-A exact Fourier series -- no lattice at all).  The oracle puts the
    ridge on ``[0, duty)`` and the cell path centres it, which is a rigid
    lateral translation and therefore efficiency-identical (locked by
    :func:`test_w8_rigid_translation_leaves_efficiencies_invariant`)."""
    o, R, T = rcwa_efficiency_1d(_P, _N_RIDGE, _N_GROOVE, 1.0, 1.0, _D, _DUTY,
                                 _WL, theta=theta, polarization=pol.lower(),
                                 n_orders=_M, formulation=formulation)
    o = np.asarray(o)
    i0 = int(np.where(o == 0)[0][0])
    i1 = int(np.where(o == 1)[0][0])
    return np.array([np.asarray(R)[i0], np.asarray(T)[i0],
                     np.asarray(R)[i1], np.asarray(T)[i1]])


# ===========================================================================
# W8-1 -- BOUNDARY COINCIDENCE (FOUND + FIXED)
# ===========================================================================

def test_w8_boundary_coincidence_keeps_the_pixel():
    """FOUND+FIXED, the headline.  ``add_tapered_grating(shear=0.5)`` at
    ``duty=0.5`` puts the ridge's lower wall at ``lo = (k + 0.5)/(2 n_slices)``
    for slice ``k``, which lands EXACTLY on the pixel centre
    ``(i + 0.5)/n_x`` for EVERY slice when ``n_x == 2 n_slices``.  The old
    symmetric ``|dist| < half`` excluded both walls, so every slice lost one
    pixel: measured pre-fix, all 128 slices realised duty
    ``0.49609375 = 127/256`` instead of ``0.5`` (error -3.90625e-03).  Post-fix
    the half-open convention realises 0.5 EXACTLY at ``n_x = 256``."""
    d = _realised_duties(n_slices=128, n_x=256, shear=0.5)
    assert d.shape == (128,)
    assert np.all(d == 0.5), (
        f"realised duty min {d.min():.9f} max {d.max():.9f} "
        f"(pre-fix: 0.49609375 on all 128 slices)")
    # the same family at (n_x=512, n_slices=256) and (n_x=512, shear=0.25,
    # n_slices=128) -- pre-fix 0.498046875 / 0.499023437
    assert np.all(_realised_duties(n_slices=128, n_x=512, shear=0.25) == 0.5)
    assert np.all(_realised_duties(n_slices=256, n_x=512, shear=0.5) == 0.5)


def test_w8_non_coincident_shears_were_already_exact_and_stay_exact():
    """The control arm: the SAME sweep at shears / ``n_x`` with no coincidence
    realised 0.5 exactly before the fix too.  This is what makes the defect a
    silent "convergence outlier" rather than a visible bug."""
    for n_x in (256, 512, 1024):
        for shear in (0.0, 1.0):
            assert np.all(_realised_duties(n_slices=128, n_x=n_x,
                                           shear=shear) == 0.5)


def test_w8_shear_convergence_outlier_vanishes_at_the_coincident_nx():
    """FOUND+FIXED in the PHYSICS.  Sheared taper on a clean-closure case
    (``P = 1 um``, ``wl = 633 nm``, ``d = 300 nm``, ``eps_ridge = 4``,
    ``M = 7``, ``shear = 0.5``, ``duty = 0.5``): the coincidence family is
    ``n_x == 2 n_slices``, so at ``n_x = 128`` the ``n_slices`` sequence is
    smooth except at ``n_slices = 64``.  Measured
    ``(R0_TE, R0_TM, T0_TE, T0_TM)`` at ``n_x = 128``::

        n_slices   pre-fix                              post-fix
              32   0.067392 0.135313 0.168182 0.569711  identical
              64   0.070403 0.125244 0.161805 0.587247  0.067328 0.135350
                                                        0.168465 0.569385
             128   0.067270 0.135390 0.168769 0.569466  identical

    Against the ``n_x = 1024`` answer the coincident point was off by 1.802e-02
    pre-fix and is off by 1.552e-04 post-fix (the ordinary ``O(1/n_x)``
    quantisation) -- a 116x improvement AT ``n_x = 128``.  Energy closure is
    <= 2e-07 in every solve here, so no truncation-instability warning muddies
    the comparison."""
    def solve(n_slices, n_x):
        st = RCWAStack(_P, n_superstrate=1.0, n_substrate=1.0, n_orders=7)
        st.add_tapered_grating(_D, eps_ridge=4.0, eps_groove=1.0,
                               duty_bottom=0.5, duty_top=0.5,
                               n_slices=n_slices, n_x=n_x, shear=0.5)
        res = st.set_source(_WL, theta=0.0).solve()
        o, R, T = res.efficiencies()
        o = np.asarray(o)
        i0 = int(np.where(o == 0)[0][0])
        closure = float(np.max(np.abs(np.asarray(R).sum(1)
                                      + np.asarray(T).sum(1) - 1.0)))
        # measured 1.1e-10 / 3.4e-09 (32 slices) and 4.2e-08 / 2.0e-07 (64)
        assert closure < 1e-6, closure     # the lossless stack must close
        return np.array([R[1, i0], R[0, i0], T[1, i0], T[0, i0]])

    fine = solve(64, 1024)                      # no coincidence at n_x=1024
    gap = float(np.max(np.abs(solve(64, 128) - fine)))
    # pre-fix 1.802e-02; post-fix 1.552e-04 -- 1e-3 leaves cross-platform room
    assert gap < 1.0e-3, f"coincident n_x=128 vs n_x=1024 gap {gap:.3e}"
    # the n_slices=32 neighbour was NEVER coincident (its family is n_x=64), so
    # the coincident point must no longer stand out against it
    neigh = float(np.max(np.abs(solve(32, 128) - solve(32, 1024))))
    assert gap < 5.0 * neigh, (gap, neigh)


def test_w8_duty_one_has_no_stray_groove_pixel():
    """The same defect at the OTHER extreme: at ``duty = 1`` the old symmetric
    test made the pixel ANTIPODAL to the ridge centre a GROOVE whenever it
    coincided (``dist == 0.5`` exactly), i.e. an "all ridge" layer with one
    hole.  Half-open puts it in the ridge."""
    # the duty=1 coincidence family is n_x == 2 n_slices / shear (measured):
    # (0.5, 64, 128), (1.0, 128, 128), (0.25, 64, 256) all lose one pixel in
    # EVERY slice pre-fix
    for shear, ns, n_x in ((0.5, 64, 128), (1.0, 128, 128), (0.25, 64, 256)):
        d = _realised_duties(n_slices=ns, n_x=n_x, shear=shear, duty=1.0)
        assert np.all(d == 1.0), (
            f"shear={shear} n_slices={ns} n_x={n_x}: min realised duty "
            f"{d.min():.9f} (want 1.0; pre-fix {1 - 1 / n_x:.9f})")
    d0 = _realised_duties(n_slices=8, n_x=64, shear=0.0, duty=0.0)
    assert np.all(d0 == 0.0)


def test_w8_tapered_ridges_and_pillars_share_the_convention():
    """The sibling generators carry the SAME fix.  A ridge / pillar whose wall
    lands exactly on a pixel centre keeps its pixel count (pre-fix: one pixel
    short per wall pair, two for the 2-D corner)."""
    nx = 64
    # ridge: width 16/64 of the period, lower wall exactly at pixel 8's centre
    w = 16 / nx * _P
    centre = (8 + 0.5) / nx * _P + 0.5 * w
    st = RCWAStack(_P, n_orders=5)
    st.add_tapered_ridges(_D, ridges=[(centre, w, w, 4.0)], eps_groove=1.0,
                          n_slices=2, n_x=nx)
    for L in st._layers:
        col = np.real(np.asarray(L.data)[:, 0])
        assert int(np.sum(col > 2.0)) == 16, "ridge lost a pixel"
    # pillar: same in both axes -> pre-fix short by one pixel per axis
    s2 = RCWAStack(_P, period_y=_P, n_orders=5, n_orders_y=5)
    s2.add_tapered_pillars(_D, pillars=[((centre, centre), (w, w), (w, w),
                                         4.0)], eps_host=1.0, n_slices=2,
                           n_x=nx, n_y=nx)
    for L in s2._layers:
        cell = np.real(np.asarray(L.data))
        assert int(np.sum(cell > 2.0)) == 16 * 16, "pillar lost a row/column"


def test_w8_touching_ridges_do_not_trip_the_overlap_guard():
    """Two ridges that TOUCH exactly (``A`` ends where ``B`` begins) are
    disjoint under the half-open convention, so the guard must not fire -- in
    EITHER raster mode (the guard reads the HARD masks by construction).  Under
    ``'area'`` the shared boundary pixel is split by area, and the two ridges
    together still realise the exact total width."""
    nx = 64
    w = 8 / nx * _P
    c1 = 0.25 * _P
    c2 = c1 + w                       # B starts exactly where A ends
    for rst in ("hard", "area"):
        st = RCWAStack(_P, n_orders=5)
        st.add_tapered_ridges(_D, ridges=[(c1, w, w, 4.0), (c2, w, w, 9.0)],
                              eps_groove=1.0, n_slices=1, n_x=nx, raster=rst)
        col = np.real(np.asarray(st._layers[0].data)[:, 0])
        # total (eps - 1) mass is width-weighted and lattice-exact here
        assert np.isclose(col.sum(), 1.0 * nx + (3.0 + 8.0) * 8, rtol=0,
                          atol=1e-12), f"{rst}: {col.sum()}"
    # a genuine overlap still raises, in both modes
    for rst in ("hard", "area"):
        with pytest.raises(ValueError, match="ridges overlap"):
            RCWAStack(_P, n_orders=5).add_tapered_ridges(
                _D, ridges=[(c1, w, w, 4.0), (c1 + 0.5 * w, w, w, 9.0)],
                eps_groove=1.0, n_slices=1, n_x=nx, raster=rst)


def test_w8_bit_identical_off_coincidence():
    """NON-DISCRIMINATOR (passes pre-fix), and the point of the fix: away from
    an exact wall/pixel-centre coincidence the half-open predicate returns the
    IDENTICAL mask the old symmetric one did.  40000 random
    ``(S, width, centre)`` triples measured 0 differences; 4000 are pinned
    here, against an INDEPENDENT half-open oracle so the statement holds even
    on a tree without the library helper."""
    rng = np.random.default_rng(20260727)
    for _ in range(4000):
        S = int(rng.choice([16, 33, 64, 97, 128, 256]))
        u = (np.arange(S) + 0.5) / S
        width = float(rng.uniform(0.0, 1.0))
        centre = float(rng.uniform(-1.5, 2.5))
        new = _half_open_mask(u, centre - 0.5 * width, width)
        old = _old_symmetric_mask(u, centre, 0.5 * width)
        assert np.array_equal(new, old), (S, width, centre)


def test_w8_half_open_recovers_exactly_the_lost_pixel():
    """The discriminating half of the equivalence: ON a coincidence the two
    predicates differ by exactly ONE pixel, the half-open count is the exact
    one, and the LIBRARY helper is the half-open one."""
    for S in (16, 64, 256):
        u = (np.arange(S) + 0.5) / S
        for k in (0, 3, S // 2):
            for jw in (1, S // 4, S // 2):
                lo, width = (k + 0.5) / S, jw / S
                new = _half_open_mask(u, lo, width)
                old = _old_symmetric_mask(u, lo + 0.5 * width, 0.5 * width)
                assert int(new.sum()) == jw
                assert int(old.sum()) == jw - 1
                assert int(np.sum(new != old)) == 1
                assert np.array_equal(_cover(u, lo, width, "hard") > 0.0, new)


# ===========================================================================
# W8-2 -- PIXEL SEMANTICS (locks; the contract the fix rests on)
# ===========================================================================

def _bandlimited(x):
    """``1.7 + 0.4 cos(2pi x) - 0.3 sin(4pi x)`` -- exact for ``|m| <= 2``."""
    return 1.7 + 0.4 * np.cos(2 * np.pi * x) - 0.3 * np.sin(4 * np.pi * x)


def test_w8_pixel_is_a_node_sample_of_the_factorization():
    """NON-DISCRIMINATOR (a contract lock).  ``eps_cell[j, i]`` MEANS the value
    at the NODE ``(j Px/Sx, i Py/Sy)``: the NODE sampling of a band-limited
    profile reproduces the exact analytic convolution matrix to 1.1e-16, while
    the MIDPOINT sampling is off by 5.9e-02.  This is what
    ``RCWAResult._cell_grid_index`` samples with ``round`` (audit W7-D) and what
    the ``add_layer`` / ``rcwa_efficiency_2d`` docstrings now state."""
    S, M = 16, 2
    coef = {0: 1.7 + 0j, 1: 0.2 + 0j, -1: 0.2 + 0j, 2: 0.15j, -2: -0.15j}
    orders, _n = _harmonic_orders_2d(M, 0)
    exact = np.array([[coef.get(int(a[0]) - int(b[0]), 0j) for b in orders]
                      for a in orders])
    node = np.broadcast_to(_bandlimited(np.arange(S) / S)[:, None].astype(_C),
                           (S, 1)).copy()
    mid = np.broadcast_to(
        _bandlimited((np.arange(S) + 0.5) / S)[:, None].astype(_C),
        (S, 1)).copy()
    e_node = float(np.max(np.abs(_eps_convolution_2d(node, orders, M, 0)
                                 - exact)))
    e_mid = float(np.max(np.abs(_eps_convolution_2d(mid, orders, M, 0)
                                - exact)))
    assert e_node < 1e-14, e_node
    assert e_mid > 1e-2, e_mid
    # and the midpoint lattice is EXACTLY a half-pixel translation: the
    # per-order DFT ratio is exp(+i pi k / S)
    fn = np.fft.fft(_bandlimited(np.arange(S) / S).astype(_C)) / S
    fm = np.fft.fft(_bandlimited((np.arange(S) + 0.5) / S).astype(_C)) / S
    for k in (1, 2, -1, -2):                    # SIGNED harmonic index
        assert abs(fm[k % S] / fn[k % S] - np.exp(1j * np.pi * k / S)) < 1e-13


def test_w8_generator_lattice_is_the_pixel_midpoint():
    """NON-DISCRIMINATOR (a contract lock).  The tapered builders write the
    pixel-CENTRE lattice ``(arange(S) + 0.5)/S``, half a pixel off the node
    lattice the factorization reads -- stated in the PIXEL CELL CONTRACT block
    and pinned here so the two documents cannot drift apart."""
    nx = 64
    duty = 0.5
    st = RCWAStack(_P, n_orders=5)
    st.add_tapered_grating(_D, eps_ridge=4.0, eps_groove=1.0,
                           duty_bottom=duty, duty_top=duty, n_slices=1,
                           n_x=nx, shear=0.0)
    got = np.real(np.asarray(st._layers[0].data)[:, 0]) > 2.0
    u_mid = (np.arange(nx) + 0.5) / nx
    u_node = np.arange(nx) / nx
    assert np.array_equal(got, _half_open_mask(u_mid, 0.5 - 0.5 * duty, duty))
    # the two lattices are genuinely different rasterizations (this duty is
    # chosen so they differ) -- pick one that discriminates
    duty2 = 0.37
    st2 = RCWAStack(_P, n_orders=5)
    st2.add_tapered_grating(_D, eps_ridge=4.0, eps_groove=1.0,
                            duty_bottom=duty2, duty_top=duty2, n_slices=1,
                            n_x=nx, shear=0.0)
    got2 = np.real(np.asarray(st2._layers[0].data)[:, 0]) > 2.0
    w2_mid = _half_open_mask(u_mid, 0.5 - 0.5 * duty2, duty2)
    w2_node = _half_open_mask(u_node, 0.5 - 0.5 * duty2, duty2)
    assert np.array_equal(got2, w2_mid)
    assert not np.array_equal(w2_mid, w2_node)


def test_w8_rigid_translation_leaves_efficiencies_invariant():
    """NON-DISCRIMINATOR (a contract lock), and the reason the half-pixel
    generator offset is harmless for a BAND-LIMITED cell: a rigid lateral
    translation of the structure only rephases the order amplitudes.  Measured
    on the band-limited profile at ``S = 64``, ``M = 9``: shifting by half a
    pixel, a whole pixel, or an arbitrary ``0.137 P`` moves every ``R``/``T``
    by <= 8.4e-15 at ``theta = 0`` and ``0.25`` rad."""
    S = 64
    node = np.arange(S) / S
    ny = 1
    base = np.broadcast_to(_bandlimited(node)[:, None].astype(_C),
                           (S, ny)).copy()
    for theta in (0.0, 0.25):
        R0, T0, _i0, _i1 = _solve_cells([base], theta=theta)
        for d in (0.5 / S, 1.0 / S, 0.137):
            cell = np.broadcast_to(
                _bandlimited(node + d)[:, None].astype(_C), (S, ny)).copy()
            R1, T1, _j0, _j1 = _solve_cells([cell], theta=theta)
            assert float(np.max(np.abs(R0 - R1))) < 1e-12
            assert float(np.max(np.abs(T0 - T1))) < 1e-12


def test_w8_hard_raster_quantises_the_width_and_area_does_not():
    """The documented ACCURACY CONSEQUENCE of a hard-assigned pixel: the
    realised feature width is the requested one rounded to the lattice,
    ``O(1/S)`` (measured up to 6.08e-02 at ``S = 16``, i.e. ``~1/S``), which no
    ``n_orders`` and no energy closure can see.  ``'area'`` realises it exactly
    (measured 3.7e-15 over 5000 random cases)."""
    rng = np.random.default_rng(8080)
    worst_hard = worst_area = 0.0
    for _ in range(1500):
        S = int(rng.choice([16, 33, 64, 97, 128, 256]))
        u = (np.arange(S) + 0.5) / S
        width = float(rng.uniform(0.0, 1.0))
        lo = float(rng.uniform(-2.0, 3.0))
        h = _cover(u, lo, width, "hard")
        a = _cover(u, lo, width, "area")
        assert np.all((h == 0.0) | (h == 1.0))
        assert np.all((a >= 0.0) & (a <= 1.0))
        worst_hard = max(worst_hard, abs(h.sum() / S - width))
        worst_area = max(worst_area, abs(a.sum() / S - width))
    assert worst_area < 1e-13, worst_area
    assert worst_hard > 1e-2, worst_hard


def test_w8_pmm_tapered_siblings_are_exempt():
    """NON-DISCRIMINATOR (an exemption lock).  The PMM tapered helpers emit
    exact ``(width_fraction, eps)`` SEGMENTS, not pixels, and resolve a strip's
    material at the strip MIDPOINT (``lo <= mid < hi``), which can never
    coincide with a wall -- so the boundary-coincidence class cannot arise and
    they were deliberately left untouched.  Pinned by reading the realised duty
    out of the stored segments at the SAME coincidence-prone parameters that
    broke the RCWA raster."""
    st = PMMStack(_P, n_substrate=1.0, degree=6)
    st.add_tapered_grating(_D, eps_ridge=4.0, eps_groove=1.0, duty_bottom=0.5,
                           duty_top=0.5, n_slices=128)
    assert len(st._layers) == 128
    for _t, segs, _slant in st._layers:
        ridge = sum(float(w) for w, e in segs
                    if abs(complex(np.asarray(e).reshape(-1)[0]) - 4.0) < 1e-12)
        assert ridge == 0.5, ridge
    # and the multi-ridge sibling's half-open midpoint resolution
    segs = PMMStack._ridges_to_segments(1.0, [(0.25, 0.5, 4.0)], 1.0)
    assert sum(float(w) for w, e in segs if complex(e) == 4.0) == 0.5


# ===========================================================================
# W8-3 -- OPT-IN AREA WEIGHTING (default OFF, bit-preserved)
# ===========================================================================

def test_w8_raster_default_is_hard_and_bit_preserved():
    """``raster`` defaults to ``'hard'`` in all three builders and the default
    call is BIT-identical to the explicit ``'hard'`` call, while ``'area'``
    genuinely differs (gray pixels appear)."""
    kw = dict(eps_ridge=4.0, eps_groove=1.0, duty_bottom=_DUTY,
              duty_top=_DUTY, n_slices=2, n_x=64, shear=0.3)
    a = RCWAStack(_P, n_orders=5).add_tapered_grating(_D, **kw)._layers
    b = RCWAStack(_P, n_orders=5).add_tapered_grating(
        _D, raster="hard", **kw)._layers
    c = RCWAStack(_P, n_orders=5).add_tapered_grating(
        _D, raster="area", **kw)._layers
    for la_, lb, lc in zip(a, b, c):
        assert np.array_equal(np.asarray(la_.data), np.asarray(lb.data))
        assert not np.array_equal(np.asarray(la_.data), np.asarray(lc.data))
        # hard is two-valued; area introduces exactly the boundary greys
        assert set(np.unique(np.real(np.asarray(la_.data)))) == {1.0, 4.0}
        grey = np.unique(np.real(np.asarray(lc.data)))
        assert len(grey) > 2 and grey.min() == 1.0 and grey.max() == 4.0

    # the sibling builders too
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


def test_w8_raster_mode_is_validated():
    """An unknown mode raises with both legal spellings named."""
    with pytest.raises(ValueError, match="raster must be 'hard'"):
        RCWAStack(_P, n_orders=5).add_tapered_grating(
            _D, eps_ridge=4.0, eps_groove=1.0, duty_bottom=0.5,
            raster="antialias")
    with pytest.raises(ValueError, match="raster must be 'hard'"):
        RCWAStack(_P, n_orders=5).add_tapered_ridges(
            _D, ridges=[(0.3e-6, 0.2e-6, 0.2e-6, 4.0)], eps_groove=1.0,
            raster="smooth")
    with pytest.raises(ValueError, match="raster must be 'hard'"):
        RCWAStack(_P, period_y=_P, n_orders=5, n_orders_y=5
                  ).add_tapered_pillars(
            _D, pillars=[((0.5e-6, 0.5e-6), (0.3e-6, 0.3e-6),
                          (0.3e-6, 0.3e-6), 4.0)], eps_host=1.0, raster="AREA2")
    # the mode is case-insensitive, like ``formulation``
    RCWAStack(_P, n_orders=5).add_tapered_grating(
        _D, eps_ridge=4.0, eps_groove=1.0, duty_bottom=0.5, raster="AREA",
        n_slices=1, n_x=64)


def test_w8_area_pillar_weight_is_the_separable_product():
    """A rectangle is SEPARABLE, so the 2-D area weight must be the exact
    product of the two per-axis coverages -- including the four CORNER pixels,
    which a naive "boundary pixels get 0.5" scheme gets wrong."""
    nx = ny = 32
    w = 0.3 * _P + 0.5 * _P / nx        # deliberately off-lattice
    c = 0.5 * _P + 0.13 * _P / nx
    st = RCWAStack(_P, period_y=_P, n_orders=5, n_orders_y=5)
    st.add_tapered_pillars(_D, pillars=[((c, c), (w, w), (w, w), 4.0)],
                           eps_host=1.0, n_slices=1, n_x=nx, n_y=ny,
                           raster="area")
    cell = np.real(np.asarray(st._layers[0].data))
    ax = _cover((np.arange(nx) + 0.5) / nx,
                          (c - 0.5 * w) / _P, w / _P, "area")
    want = 1.0 + 3.0 * ax[:, None] * ax[None, :]
    assert np.max(np.abs(cell - want)) < 1e-14
    # the total (eps - 1) mass equals the exact pillar AREA fraction
    assert abs((cell.sum() - nx * ny) / (3.0 * nx * ny)
               - (w / _P) ** 2) < 1e-13


@pytest.mark.parametrize("pol", ["TE", "TM"])
def test_w8_area_beats_hard_on_the_default_laurent_rule(pol):
    """MEASURED VERDICT, arm 1.  Against the EXACT analytic 1-D oracle
    (vertical ``duty = 0.37``, ``n = 2/1``, ``P = 1 um``, ``wl = 633 nm``,
    ``d = 300 nm``, ``M = 9``, ``theta = 0.25`` rad) ``'area'`` is 1-3 orders of
    magnitude more accurate than ``'hard'`` at the SAME ``n_x`` for BOTH
    polarizations under the default ``formulation='laurent'``.  Measured
    (max over ``R0, T0, R+1, T+1``)::

        n_x     TE hard    TE area    TM hard    TM area
         64    5.49e-03   2.86e-04   2.33e-03   5.67e-05
        256    3.10e-03   5.56e-05   1.34e-03   1.51e-05
       1024    9.29e-04   3.84e-06   4.02e-04   1.03e-06
    """
    exact = _quad_exact(pol, "laurent", 0.25)
    prev_area = None
    for n_x, floor in ((64, 4.0), (256, 8.0), (1024, 20.0)):
        e_h = float(np.max(np.abs(
            _quad(_cell_of(n_x=n_x, raster="hard"), pol, theta=0.25) - exact)))
        e_a = float(np.max(np.abs(
            _quad(_cell_of(n_x=n_x, raster="area"), pol, theta=0.25) - exact)))
        assert e_a * floor < e_h, (n_x, pol, e_h, e_a)
        if prev_area is not None:
            assert e_a < prev_area          # and it actually converges
        prev_area = e_a
    assert prev_area < 1e-5, prev_area


def test_w8_area_hurts_the_li_wall_normal_polarization():
    """MEASURED VERDICT, arm 2 -- the reason ``'area'`` ships OPT-IN and
    DEFAULT OFF.  Li's inverse rule assumes a SHARP interface, and the
    ARITHMETIC (area) average is the wrong effective medium for the
    wall-NORMAL component -- it wants the HARMONIC one (Farjadpour 2006
    subpixel smoothing).  On the SHEARED taper (``shear = 0.4``,
    ``duty = 0.37``, 16 slices, ``M = 9``, reference ``n_x = 16384``)
    ``formulation='li'`` + TM measured ``'area'`` WORSE than ``'hard'`` by
    10.6x at ``n_x = 64``, 1.2x at 128, 9.2x at 256 and 2.2x at 512, while TE on
    the same sweep improved by up to 120x.  On the vertical grating vs the exact
    oracle ``'li'`` TM ``'area'`` gains only ~2.5x and PLATEAUS (2.2e-05 at
    ``n_x = 8192`` against 1.6e-08 for ``'laurent'`` TM)."""
    kw = dict(n_slices=16, shear=0.4)
    ref = _quad(_cell_of(n_x=16384, raster="area", **kw), "TM",
                formulation="li")
    e_h = float(np.max(np.abs(
        _quad(_cell_of(n_x=64, raster="hard", **kw), "TM",
              formulation="li") - ref)))
    e_a = float(np.max(np.abs(
        _quad(_cell_of(n_x=64, raster="area", **kw), "TM",
              formulation="li") - ref)))
    assert e_a > 3.0 * e_h, ("li/TM: area was expected WORSE than hard",
                             e_h, e_a)
    # ... while TE on the very same cells improves by an order of magnitude
    t_h = float(np.max(np.abs(
        _quad(_cell_of(n_x=64, raster="hard", **kw), "TE",
              formulation="li")
        - _quad(_cell_of(n_x=16384, raster="area", **kw), "TE",
                formulation="li"))))
    t_a = float(np.max(np.abs(
        _quad(_cell_of(n_x=64, raster="area", **kw), "TE",
              formulation="li")
        - _quad(_cell_of(n_x=16384, raster="area", **kw), "TE",
                formulation="li"))))
    assert t_a < t_h, (t_h, t_a)


def test_w8_te_is_formulation_independent_for_a_1d_cell():
    """The measurement that makes the per-polarization recommendation legible:
    for a 1-D (x-only) cell ``E_y`` is TANGENTIAL to every wall, so Li's rule
    reduces to the direct rule and the TE row is BIT-identical between
    ``'laurent'`` and ``'li'`` (measured 0.0) -- while the TM row differs by
    ~1.3e-03 to 1.7e-03.  So ``'area'``'s TE verdict cannot depend on the
    formulation, and only the TM arm needed deciding."""
    for rst in ("hard", "area"):
        for n_x in (64, 256):
            cells = _cell_of(n_x=n_x, raster=rst)
            Rl, _Tl, _i, _j = _solve_cells(cells, formulation="laurent",
                                           theta=0.25)
            Ri, _Ti, _k, _m = _solve_cells(cells, formulation="li", theta=0.25)
            assert np.array_equal(Rl[_ROW["TE"]], Ri[_ROW["TE"]])
            assert float(np.max(np.abs(Rl[_ROW["TM"]] - Ri[_ROW["TM"]]))) > 1e-4


def test_w8_area_converges_to_the_same_limit_as_hard():
    """``'area'`` changes the DISCRETISATION, not the structure: at large
    ``n_x`` both modes must agree, so ``'area'`` is a convergence accelerator
    and not a different model.  Measured on the sheared taper at
    ``n_x = 16384``: 2.44e-05 (the residual staircase-independent quantisation),
    against 2.4e-02 at ``n_x = 37``."""
    kw = dict(n_slices=8, shear=0.4)
    gaps = []
    for n_x in (64, 2048, 16384):
        h = _quad(_cell_of(n_x=n_x, raster="hard", **kw), "TE")
        a = _quad(_cell_of(n_x=n_x, raster="area", **kw), "TE")
        gaps.append(float(np.max(np.abs(h - a))))
    assert gaps[-1] < 1e-3, gaps
    assert gaps[-1] < 0.2 * gaps[0], gaps


# ===========================================================================
# plot_geometry -- the same convention on the DISPLAY raster
# ===========================================================================

def test_w8_plot_geometry_renders_a_wrapping_rectangle():
    """The shapes branch of :meth:`RCWAStack.plot_geometry` rasterized with
    ``|xs - cx| < wx/2``: not wrap-aware, so a rectangle CROSSING the cell edge
    vanished from the picture even though the solver's analytic form factor
    included it ("the picture IS the model" broken).  Now half-open and
    wrap-aware."""
    st = RCWAStack(_P, period_y=_P, n_orders=5, n_orders_y=5)
    st.add_layer(_D, eps_background=1.0, shapes=[
        {"shape": "rectangle", "center": (0.98 * _P, 0.5 * _P),
         "size": (0.3 * _P, 0.6 * _P), "eps": 4.0}])
    ax = st.plot_geometry()
    row = np.asarray(ax.get_images()[0].get_array()).ravel()
    assert row.max() > 3.9, "the wrapping rectangle did not render at all"
    n_hi = int(np.sum(row > 3.9))
    assert abs(n_hi - 0.3 * len(row)) <= 1, n_hi
    # and it renders on BOTH sides of the cell edge (that is the wrap)
    assert row[0] > 3.9 and row[-1] > 3.9
