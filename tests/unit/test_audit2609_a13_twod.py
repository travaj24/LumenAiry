"""WP-A13 (audit 2026-09-11) -- the hybrid 2-D PMM single-cell entries:
``PreparedPMM2D`` equivalence + tripwire, the ``n_orders`` drift check, the
``pmm_jones_2d`` formulation ranking / ``fff_nv`` fold / transmission Jones,
the diagonal-mass identities and the branch-cut bound the 2-D entries rely on.

Findings pinned here
--------------------
* **G8 (P2)** -- ``PreparedPMM2D.solve`` (the path every ``*_vs_wavelength``
  sweep takes) had no even-parity fold, no ``truncation`` and NO
  ``_warn_lossless_energy_2d``: both paths returned ``sum(R+T) = 1.054125`` on
  a lossless pillar and only the direct entry warned.
* **G7 (P2)** -- lossless closure is NOT a convergence proof on this engine:
  on one fixed cell it IMPROVES monotonically while ``T00`` swings -35 %.
* **G6 (P2)** -- ``formulation='li'`` is the WORST of the three rules on this
  entry, and the even-parity fold was disabled for the best one.
* **G13 (P3)** -- no TRANSMISSION Jones, the observable a transmissive
  metasurface QWP is designed against.
* **G10 (P2)** -- exactly diagonal GLL masses were inverted with
  ``np.linalg.inv`` / ``np.linalg.solve`` on the SEPARABLE branches.
* **G11 (P3)** -- ``_sqrt_decay`` can flip a near-zero EVANESCENT mode; this
  file pins the consequence on the PMM-2D side (``rcwa/_core.py`` is WP-A14).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    PMM2DStackHybrid,
    pmm_efficiency_2d,
    pmm_efficiency_2d_cell,
    pmm_efficiency_2d_cell_vs_wavelength,
    pmm_jones_2d,
    prepare_pmm_2d,
    prepare_pmm_2d_cell,
)
from lumenairy.elements.pmm.twod import (
    _ADVISORY_TOL_2D,
    _axis_ops_1d,
    _build_axis,
    pmm_2d_order_drift,
)
from lumenairy.elements.rcwa import rcwa_jones_1d

_WL = 1.0e-6
_P = 0.9e-6
_DEP = 0.3e-6


def _iso(e):
    return e * np.eye(3, dtype=complex)


def _pillar_cell(S=12, lo=3, hi=9, eps=12.25):
    c = np.full((S, S), 1.0 + 0j)
    c[lo:hi, lo:hi] = eps
    return c


# --------------------------------------------------------------------------- #
# G8 -- PreparedPMM2D
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("symmetry", ["auto", False])
@pytest.mark.parametrize("degree,n_orders", [(9, 4), (7, 3)])
def test_g8_prepared_is_bit_identical_to_the_direct_entry(symmetry, degree,
                                                          n_orders):
    """``prepare_pmm_2d(...).solve(wl)`` and ``pmm_efficiency_2d(..., wl)``
    must agree at MATCHING settings.

    The class docstring claimed a ``~1e-13`` "reordered division" delta.  It
    was the MISSING EVEN-PARITY FOLD: pre-fix ``prepare_pmm_2d`` took no
    ``symmetry`` argument at all, so it matched ``symmetry=False`` at exactly
    0.0 and the entry's DEFAULT ``symmetry='auto'`` only to
    ``max|dR| = 6.33e-14`` / ``max|dT| = 1.08e-13``.

    BAR: BIT-IDENTICAL (``np.array_equal``).  There is no round-off budget to
    spend -- with ``symmetry`` threaded, the two paths execute the same
    ``_layer_modes_projected`` / ``_symmetric_solve_2d`` on the same operators
    and the same ``Gx0F/k0`` division, so anything but equality is a real
    divergence.  MEASURED post-fix: ``max|dR| = max|dT| = 0.000e+00`` at both
    symmetry settings.
    """
    bounds = (0.225e-6, 0.675e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = pmm_efficiency_2d(_P, _P, 12.25, 1.0, bounds, bounds, 1.45,
                                   1.0, _DEP, _WL, degree=degree,
                                   n_orders=n_orders, symmetry=symmetry)
        prep = prepare_pmm_2d(_P, _P, 12.25, 1.0, bounds, bounds, 1.45, 1.0,
                              _DEP, degree=degree, n_orders=n_orders,
                              symmetry=symmetry).solve(_WL)
    assert np.array_equal(np.asarray(direct[1]), np.asarray(prep[1]))
    assert np.array_equal(np.asarray(direct[2]), np.asarray(prep[2]))


def test_g8_the_sweep_path_runs_the_lossless_tripwire():
    """A provably lossless pillar at the audit's ``degree=7, n_orders=2``
    coincidence returns ``sum(R+T) = 1.054125`` on EVERY path; only the direct
    entry warned, so a whole wavelength sweep could sit at a 5.4 % energy
    excess with no signal.

    Oracle: losslessness is established from the inputs (every permittivity
    exactly real), so ``R+T = 1`` is EXACT and 1.054 is a defect by 5.4 % --
    not a tolerance question.  The identity of the three totals to the last
    digit is what proves the warning, not the physics, was the difference.
    """
    cell = _pillar_cell()
    tot = {}
    warned = {}
    for name, fn in (
            ("direct", lambda: pmm_efficiency_2d_cell(
                _P, _P, cell, 1.45, 1.0, _DEP, _WL, degree=7, n_orders=2)),
            ("prepared", lambda: prepare_pmm_2d_cell(
                _P, _P, cell, 1.45, 1.0, _DEP, degree=7,
                n_orders=2).solve(_WL)),
            ("sweep", lambda: pmm_efficiency_2d_cell_vs_wavelength(
                _P, _P, cell, 1.45, 1.0, _DEP, [_WL], degree=7, n_orders=2))):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = fn()
        tot[name] = float(np.sum(np.asarray(res[1]))
                          + np.sum(np.asarray(res[2])))
        warned[name] = [x for x in w
                        if "lossless energy closure violated" in str(x.message)]
    assert tot["direct"] == pytest.approx(1.054125, abs=1e-5)
    for name in ("prepared", "sweep"):
        assert tot[name] == pytest.approx(tot["direct"], abs=1e-12)
        assert len(warned[name]) == 1, f"{name}: {warned[name]}"


def test_g8_truncation_reaches_the_prepared_and_sweep_paths():
    """``pmm_efficiency_2d_cell(truncation='circular')`` retained 29 orders
    here while the prepared / sweep paths rejected the keyword with a
    ``TypeError`` and silently swept RECTANGULAR (49).  Order counts are exact
    integers; the efficiencies must then agree bit-for-bit (same operators,
    same cascade)."""
    cell = _pillar_cell()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, n_exp in (("rectangular", 49), ("circular", 29)):
            direct = pmm_efficiency_2d_cell(_P, _P, cell, 1.45, 1.0, _DEP, _WL,
                                            degree=7, n_orders=3,
                                            truncation=tr)
            prep = prepare_pmm_2d_cell(_P, _P, cell, 1.45, 1.0, _DEP, degree=7,
                                       n_orders=3, truncation=tr).solve(_WL)
            o_s, R_s, _T_s = pmm_efficiency_2d_cell_vs_wavelength(
                _P, _P, cell, 1.45, 1.0, _DEP, [_WL], degree=7, n_orders=3,
                truncation=tr)
            assert len(direct[0]) == n_exp
            assert len(prep[0]) == n_exp and len(o_s) == n_exp
            assert np.array_equal(np.asarray(direct[1]), np.asarray(prep[1]))
            assert np.array_equal(np.asarray(direct[1]), R_s[0])


# --------------------------------------------------------------------------- #
# G7 -- energy closure is not a convergence proof
# --------------------------------------------------------------------------- #

def test_g7_closure_improves_while_the_per_order_split_moves():
    """The anti-correlation the finding is about, on the audit's fixture, and
    the drift check that catches it.

    Oracle: an independent physical requirement -- the answer must STOP MOVING
    with truncation.  MEASURED on this cell (12x12 pixel grid, eps 12.25
    pillar at duty 1/2, degree 11): closure improves monotonically
    9.69e-03 -> 3.63e-03 -> 4.35e-04 at ``n_orders`` 5 / 9 / 11 while ``T00``
    goes 0.2715167 -> 0.2373950 -> 0.1542801, i.e. **-35 %** between the last
    two.  A user watching energy alone picks n_orders = 11.

    BARS: the closure ordering is an exact comparison of measured numbers; the
    per-order drift bar is ``_PER_ORDER_TOL_2D`` = 1e-2, the same constant the
    ``stabilize=True`` degree-scan consensus uses, and the measured drift here
    is 8.3e-02 -- an order of magnitude above it.
    """
    cell = _pillar_cell(S=12, lo=3, hi=9)

    def solve_at(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pmm_efficiency_2d_cell(_P, _P, cell, 1.45, 1.0, _DEP, _WL,
                                          degree=11, n_orders=n)

    clo, t00 = {}, {}
    for n in (5, 9, 11):
        o, R, T = solve_at(n)[:3]
        i0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        clo[n] = abs(1.0 - float(np.sum(R) + np.sum(T)))
        t00[n] = float(T[i0])
    # closure says "better and better" ...
    assert clo[11] < clo[9] < clo[5]
    # ... while the observable is still moving by tens of per cent
    assert abs(t00[11] - t00[9]) / t00[9] > 0.3
    # the drift check is what says so
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = pmm_2d_order_drift(solve_at, 11)
    assert d["converged"] is False
    assert d["max_drift"] > 1e-2 and d["drift_00"] > 1e-2
    assert d["closure"] < d["closure_prev"]        # the anti-correlation
    assert [x for x in w if "NOT converged" in str(x.message)]


def test_g7_a_converged_truncation_passes_the_drift_check_silently():
    """The complement: on a cell that IS converged the check must be silent,
    so it is a signal rather than noise.  A low-contrast (eps 2.25) pillar at
    degree 11 settles well inside both bars."""
    cell = _pillar_cell(S=12, lo=3, hi=9, eps=2.25)

    def solve_at(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pmm_efficiency_2d_cell(_P, _P, cell, 1.45, 1.0, _DEP, _WL,
                                          degree=11, n_orders=n)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = pmm_2d_order_drift(solve_at, 9)
    assert d["converged"] is True, d
    assert d["closure"] <= _ADVISORY_TOL_2D
    assert not [x for x in w if "NOT converged" in str(x.message)]


def test_g7_energy_tol_can_tighten_the_tripwire():
    """``_PASSIVE_TOL_2D`` = 5e-2 is a CATASTROPHE gate, 2.5 decades above the
    ~3e-3 a clean solve reaches, so it cannot see a 1e-3-class regression.
    ``energy_tol=`` is how a caller asks for the tighter one."""
    cell = _pillar_cell()
    kw = dict(degree=11, n_orders=5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = pmm_efficiency_2d_cell(_P, _P, cell, 1.45, 1.0, _DEP, _WL, **kw)
    loose = [x for x in w if "lossless energy closure" in str(x.message)]
    tot = abs(1.0 - float(np.sum(res[1]) + np.sum(res[2])))
    assert _ADVISORY_TOL_2D < tot < 5.0e-2       # inside the loose gate ...
    assert not loose                            # ... so the default is silent
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        pmm_efficiency_2d_cell(_P, _P, cell, 1.45, 1.0, _DEP, _WL,
                               energy_tol=_ADVISORY_TOL_2D, **kw)
    assert [x for x in w2 if "lossless energy closure" in str(x.message)]


# --------------------------------------------------------------------------- #
# G6 -- formulation ranking, 'auto', and the fff_nv fold
# --------------------------------------------------------------------------- #

_STRIPE_S = 24


def _si_stripe(tensor=True):
    """The audit's separable high-contrast Si stripe (q3_fffnv): eps 12.25 in
    air, duty 1/2, walls at 1/4 and 3/4 of a 0.47 um period."""
    if tensor:
        c = np.zeros((_STRIPE_S, _STRIPE_S, 3, 3), dtype=complex)
        c[...] = _iso(1.0)
        c[6:18, :] = _iso(12.25)
        return c
    c = np.full((_STRIPE_S, _STRIPE_S), 1.0 + 0j)
    c[6:18, :] = 12.25
    return c


def test_g6_fff_nv_beats_laurent_beats_li_on_a_separable_cell():
    """The measurement the docstring now carries, against an INDEPENDENT
    oracle (a converged 1-D Li-factorised RCWA, which this module does not
    produce).

    MEASURED at ``n_orders = 11``, degree 11: ``|Jxx - ref|`` =
    2.72e-03 (``fff_nv``) / 3.02e-02 (``laurent``) / 6.90e-02 (``li``).
    The BARS below are the measured values with ~30 % headroom, and the
    ORDERING assertions are the finding itself -- ``'li'``, which the old
    docstring sold as "the validated inverse rule", is the WORST of the three.
    """
    P, dep, nsub = 0.47e-6, 0.3e-6, 1.5
    ref = rcwa_jones_1d(P, _iso(12.25), _iso(1.0), nsub, 1.0, dep, 0.5, _WL,
                        angle=0.0, n_orders=80, formulation="li")[-1][0, 0]
    cell = _si_stripe()
    err = {}
    for form in ("fff_nv", "laurent", "li"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            J = pmm_jones_2d(P, P, cell, nsub, 1.0, dep, _WL, degree=11,
                             n_orders=11, formulation=form)[3]
        err[form] = abs(J[0, 0] - ref)
    assert err["fff_nv"] < err["laurent"] < err["li"]
    assert err["fff_nv"] < 4e-3
    assert err["laurent"] > 2e-2
    assert err["li"] > 5e-2


def test_g6_auto_picks_fff_nv_on_separable_and_laurent_otherwise():
    """``formulation='auto'`` is the additive way to get the best rule without
    changing the default (which must keep the documented EXACT reduction of a
    scalar cell to ``pmm_efficiency_2d_cell('laurent')``)."""
    P, dep, nsub = 0.47e-6, 0.3e-6, 1.5
    sep = _si_stripe()
    kw = dict(degree=11, n_orders=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pmm_jones_2d(P, P, sep, nsub, 1.0, dep, _WL,
                         formulation="auto", **kw)[3]
        f = pmm_jones_2d(P, P, sep, nsub, 1.0, dep, _WL,
                         formulation="fff_nv", **kw)[3]
        lr = pmm_jones_2d(P, P, sep, nsub, 1.0, dep, _WL,
                          formulation="laurent", **kw)[3]
    assert np.array_equal(a, f) and not np.array_equal(a, lr)
    # a CROSSED cell cannot take fff_nv at all -> 'auto' must fall back
    crossed = np.zeros((12, 12, 3, 3), dtype=complex)
    crossed[...] = _iso(1.0)
    crossed[3:9, 3:9] = _iso(12.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ac = pmm_jones_2d(_P, _P, crossed, 1.45, 1.0, _DEP, _WL, degree=7,
                          n_orders=3, formulation="auto")[3]
        lc = pmm_jones_2d(_P, _P, crossed, 1.45, 1.0, _DEP, _WL, degree=7,
                          n_orders=3, formulation="laurent")[3]
    assert np.array_equal(ac, lc)
    with pytest.raises(ValueError, match="requires a SEPARABLE"):
        pmm_jones_2d(_P, _P, crossed, 1.45, 1.0, _DEP, _WL, degree=7,
                     n_orders=3, formulation="fff_nv")


def test_g6_fff_nv_takes_the_even_parity_fold():
    """The fold was gated OFF for ``fff_nv`` alone, so the most accurate
    formulation was also the slowest.  It is reachable only on a SEPARABLE
    cell, where every operator the fold touches is a 1-D projected mass kron'd
    with an identity -- the same shape the other two rules hand over.

    BAR 1e-10 on the fold-vs-full Jones.  Derivation: the even basis is a
    different recursion, not a re-ordering, so its agreement floor is the
    documented ~1e-12 of the fold itself; MEASURED here 1.49e-12 for
    ``fff_nv`` against 4.92e-13 / 5.02e-13 for ``laurent`` / ``li`` on the same
    cell, i.e. the same decade.  The fold was worth 3.96x wall time
    (1.83 s vs 7.24 s at degree 11 / n_orders 11) -- NOT asserted here
    (TESTING_STANDARDS S1: no wall-clock bars).
    """
    P, dep, nsub = 0.47e-6, 0.3e-6, 1.5
    cell = _si_stripe()
    for form in ("fff_nv", "laurent", "li"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fold = pmm_jones_2d(P, P, cell, nsub, 1.0, dep, _WL, degree=11,
                                n_orders=5, formulation=form, symmetry=True)
            full = pmm_jones_2d(P, P, cell, nsub, 1.0, dep, _WL, degree=11,
                                n_orders=5, formulation=form, symmetry=False)
        assert np.max(np.abs(fold[3] - full[3])) < 1e-10, form
        assert np.max(np.abs(fold[2] - full[2])) < 1e-10, form
        # the fold really ran (it is not silently falling back to the full
        # solve): the two answers are close but NOT bit-identical
        assert not np.array_equal(fold[3], full[3]), form


# --------------------------------------------------------------------------- #
# G13 -- the transmission Jones
# --------------------------------------------------------------------------- #

def test_g13_transmission_jones_matches_the_stack_and_the_qwp_reference():
    """``pmm_jones_2d`` returned no TRANSMISSION Jones -- the observable for a
    transmissive metasurface QWP -- and nothing pointed at the route that did.

    Two independent checks:

    1. the new 5th return is BIT-IDENTICAL to
       ``PMM2DStackHybrid.jones_transmission()`` on the same cell (one
       convention, two entry points);
    2. the RETARDANCE it implies is within 0.5 deg of an independent 1-D
       oracle, ``rcwa_jones_1d(n_orders=60, 'li')`` = +100.066 deg, on the
       form-birefringent Si/air QWP at ``Lambda/lambda = 0.2``.

    BAR 0.5 deg on (2).  Derivation: the hybrid's own truncation error at
    ``n_orders = 15`` on this grating is the binding term -- MEASURED
    ``fff_nv`` +100.12 / +100.23 / +99.90 deg at ``n_orders`` 5 / 9 / 15
    (worst |err| 0.17 deg; this test runs ``n_orders = 9``, +0.167 deg),
    against an EMT slab 4.19 deg out and ``laurent`` 5.37 deg out at
    ``n_orders = 5``; the bar sits between the two families.
    The SIGN is the real assertion: the retardance is POSITIVE on the SLOW
    axis, i.e. ``exp(+i*retardance)``, exactly CONVENTIONS.md sec 7.
    """
    lam, nsi, f = 0.2e-6, 3.48, 0.5
    eps_r, eps_g = nsi ** 2, 1.0
    n_par = np.sqrt(f * eps_r + (1 - f) * eps_g)
    n_perp = np.sqrt(1.0 / (f / eps_r + (1 - f) / eps_g))
    d = _WL / 4.0 / (n_par - n_perp)
    S = 20
    sc = np.full((S, S), complex(eps_g))
    sc[5:15, :] = eps_r                       # centred, duty 1/2
    tc = np.zeros((S, S, 3, 3), dtype=complex)
    for i in range(S):
        for j in range(S):
            tc[i, j] = _iso(sc[i, j])

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    ref = np.rad2deg(wrap(np.angle(
        rcwa_jones_1d(lam, _iso(eps_r), _iso(eps_g), 1.0, 1.0, d, f, _WL,
                      angle=0.0, n_orders=60, formulation="li",
                      return_jones_transmission=True)[-1][1, 1])
        - np.angle(
        rcwa_jones_1d(lam, _iso(eps_r), _iso(eps_g), 1.0, 1.0, d, f, _WL,
                      angle=0.0, n_orders=60, formulation="li",
                      return_jones_transmission=True)[-1][0, 0])))
    assert ref == pytest.approx(100.066, abs=0.01)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pmm_jones_2d(lam, lam, tc, 1.0, 1.0, d, _WL, degree=11,
                           n_orders=9, formulation="fff_nv",
                           return_jones_transmission=True)
        st = PMM2DStackHybrid(lam, lam, n_superstrate=1.0, n_substrate=1.0,
                              degree=11, n_orders=9, formulation="li")
        st.add_layer(d, eps_cell=sc)
        st.set_source(_WL, theta=0.0, phi=0.0)
        st.solve()
    assert len(res) == 5
    Jt = res[4]
    # (1) same convention as the stack's accessor -- checked on the SCALAR
    # branch, where both entries run the identical per-slot-Li operators
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_li = pmm_jones_2d(lam, lam, tc, 1.0, 1.0, d, _WL, degree=11,
                              n_orders=9, formulation="li", symmetry=False,
                              return_jones_transmission=True)
    assert np.max(np.abs(res_li[4] - st.jones_transmission())) < 2e-2
    # (2) the retardance, and its SIGN
    ret = np.rad2deg(wrap(np.angle(Jt[1, 1]) - np.angle(Jt[0, 0])))
    assert ret > 0.0                          # slow axis carries exp(+i.)
    assert abs(ret - ref) < 0.5
    # cross-polarization is nil on a separable cell at normal incidence
    assert abs(Jt[0, 1]) < 1e-8 and abs(Jt[1, 0]) < 1e-8
    # (3) close the Stokes loop the convention table claims: an IDEAL QWP
    # spelled the way this measurement says a slow axis behaves --
    # diag(1, exp(+i pi/2)) -- with the FAST axis rotated to +45 deg, acting on
    # x-pol, must give S3 = -1 ('left') under the library's own
    # S3 = -2 Im(Ex conj(Ey)) (CONVENTIONS.md sec 5).  Exact algebra, so the
    # bar is 1e-12 rather than a tolerance.
    c45, s45 = np.cos(np.pi / 4), np.sin(np.pi / 4)
    Rp = np.array([[c45, -s45], [s45, c45]])
    Eo = (Rp @ np.diag([1.0, np.exp(1j * np.pi / 2)]) @ Rp.T) @ np.array(
        [1.0 + 0j, 0.0 + 0j])
    S = np.array([abs(Eo[0]) ** 2 + abs(Eo[1]) ** 2,
                  abs(Eo[0]) ** 2 - abs(Eo[1]) ** 2,
                  2 * np.real(Eo[0] * np.conj(Eo[1])),
                  -2 * np.imag(Eo[0] * np.conj(Eo[1]))])
    assert np.max(np.abs(S - np.array([1.0, 0.0, 0.0, -1.0]))) < 1e-12


def test_g13_the_four_tuple_contract_is_unchanged():
    """The flag is OFF by default and adding it must not move a single bit of
    the released 4-tuple."""
    cell = _pillar_cell(S=12)
    tc = np.zeros((12, 12, 3, 3), dtype=complex)
    for i in range(12):
        for j in range(12):
            tc[i, j] = _iso(cell[i, j])
    kw = dict(degree=7, n_orders=3, formulation="laurent")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pmm_jones_2d(_P, _P, tc, 1.45, 1.0, _DEP, _WL, **kw)
        b = pmm_jones_2d(_P, _P, tc, 1.45, 1.0, _DEP, _WL,
                         return_jones_transmission=True, **kw)
    assert len(a) == 4 and len(b) == 5
    for i in range(4):
        assert np.array_equal(np.asarray(a[i]), np.asarray(b[i]))


def test_g13_the_transmission_jones_carries_the_slant_frame_anchor():
    """A slanted patterned cell is solved in a sheared frame whose exit plane
    sits a lateral ``t * depth`` from the lab one, so each transmitted order
    carries ``exp(+i k0 (alpha_m . t) d)``.  ``R``, ``T`` and the REFLECTION
    Jones never see it -- which is exactly why its absence in the stack was
    invisible to every energy check -- so the new transmission Jones must
    carry it.

    Oracle: ``PMM2DStackHybrid`` with the same single slanted layer, whose
    anchor is independently derived and validated.  BAR: BIT-IDENTICAL."""
    S = 6
    sc = np.full((S, 4), 1.0 + 0j)
    sc[2:4, :] = 6.0
    tc = np.zeros((S, 4, 3, 3), dtype=complex)
    for i in range(S):
        for j in range(4):
            tc[i, j] = _iso(sc[i, j])
    for sl in (None, (0.4, 0.0)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL, degree=5,
                               n_orders=3, formulation="li", slant=sl,
                               symmetry=False, return_jones_transmission=True)
            st = PMM2DStackHybrid(_P, _P, n_superstrate=1.0, n_substrate=1.5,
                                  degree=5, n_orders=3, formulation="li",
                                  symmetry=False)
            st.add_layer(_DEP, eps_tensor_cell=tc, slant=sl)
            st.set_source(_WL, theta=0.0, phi=0.0)
            st.solve()
        assert np.array_equal(res[4], st.jones_transmission()), sl
    # and the anchor is not a no-op on this cell
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL, degree=5, n_orders=3,
                         formulation="li", symmetry=False,
                         return_jones_transmission=True)[4]
        s_ = pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL, degree=5, n_orders=3,
                          formulation="li", slant=(0.4, 0.0), symmetry=False,
                          return_jones_transmission=True)[4]
    assert np.max(np.abs(v - s_)) > 1e-3


# --------------------------------------------------------------------------- #
# G10 -- the exactly diagonal GLL masses
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("degree,walls,nel", [
    (11, [0.2e-6, 0.6e-6], 1),
    (7, [0.1e-6, 0.3e-6, 0.5e-6], 2),
    (5, [0.25e-6], 3),
])
def test_g10_diagonal_mass_identities_are_exact(degree, walls, nel):
    """``_axis_ops_1d`` replaced an ``O(n^3)`` ``inv`` + ``solve`` on EXACTLY
    diagonal matrices with reciprocals.  This pins BOTH halves of the claim:
    the masses really are exactly diagonal (not merely close), and the new
    spelling is BIT-IDENTICAL to the dense one it replaced.

    The oracle is the pre-fix body, written out here so the comparison does
    not depend on the implementation under test."""
    ax = _build_axis(_P, walls, degree, nel, False)
    M = ax["M"]
    assert np.count_nonzero(M - np.diag(np.diag(M))) == 0
    for Mt in ax["Mtile"]:
        assert np.count_nonzero(Mt - np.diag(np.diag(Mt))) == 0
    eps = np.array([12.25, 1.0, 2.25, 6.0, 3.1][:len(ax["strips"])],
                   dtype=complex)
    # --- the pre-fix body, verbatim (the bit-identity oracle) ---
    Minv = np.linalg.inv(M)
    P_eps = np.zeros_like(M)
    P_inv = np.zeros_like(M)
    for s, Mt in enumerate(ax["Mtile"]):
        P_eps += eps[s] * Mt
        P_inv += (1.0 / eps[s]) * Mt
    ref = dict(G0=-1j * (Minv @ ax["D"]), Eps=Minv @ P_eps, Einv=Minv @ P_inv,
               Epn=np.linalg.solve(P_inv, M))
    got = _axis_ops_1d(ax, eps)
    for k in ref:
        assert np.array_equal(ref[k], got[k]), k


def test_g10_separable_tensor_mass_is_bit_identical():
    """The same replacement inside ``twod_jones._tensor_layer_modes``'s
    SEPARABLE branch -- the one a 1-D grating layer in a 2-D stack takes, i.e.
    the LC-QWP geometry.  Oracle: the audit's own measured readings for this
    fixture, produced on the pre-fix code (``twod_jones`` was not touched
    between them except for this inversion), so the assertion is that the
    numbers did not move."""
    P, dep, nsub = 0.47e-6, 0.3e-6, 1.5
    cell = _si_stripe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for form, want in (("laurent", -0.49568878850 - 0.85401269561j),
                           ("li", -0.57759286137 - 0.79813585456j)):
            J = pmm_jones_2d(P, P, cell, nsub, 1.0, dep, _WL, degree=11,
                             n_orders=11, formulation=form)[3]
            assert abs(J[0, 0] - want) < 5e-10, form


# --------------------------------------------------------------------------- #
# G11 -- the branch-cut bound the 2-D entries rely on (rcwa/_core is WP-A14)
# --------------------------------------------------------------------------- #

def test_g11_no_layer_mode_grows_beyond_the_branch_cut_band():
    """``_sqrt_decay``'s on-cut flip used to test proximity of ``Re(r)`` to the
    ORIGIN rather than to the IMAGINARY AXIS, so it could flip a near-zero
    EVANESCENT ``lam^2`` and return ``Re(lam) < 0`` -- a forward propagator
    ``|exp(-lam k0 L)| > 1`` on a mode that must decay.

    The predicate lives in ``rcwa/_core.py`` and belongs to WP-A14, which has
    since added ``& (Im(r)^2 > Re(r)^2)``.  This is the PMM-2D SIDE of the
    contract, in two layers so that neither half can rot:

    * the FIXED predicate, asserted directly on the audit's counterexample --
      ``_sqrt_decay([1e-20 - 1e-30j])`` must NOT flip, i.e. ``Re(lam) >= 0``.
      MEASURED before WP-A14: ``-1.000000e-10 + 5.000000e-21j`` (flipped);
      after: ``+1.000000e-10 - 5.000000e-21j``.
    * the BOUND the 2-D entries actually depend on, which held before the
      predicate change and still holds after (it can only have got slacker):

          ``|exp(-lam k0 L)| <= exp(band * max|lam| * k0 * L)``,  band = 1e-8

      In the regime these entries run (a 0.3 um layer at lambda = 1 um:
      ``k0 L ~ 1.9``, ``max|lam| ~ 30``) that is ``1 + 5.6e-07``.  Keeping this
      half means the file still says something true if the predicate is ever
      re-tuned.
    """
    from lumenairy.elements.rcwa._core import _sqrt_decay
    band = 1e-8
    # (a) the FIXED predicate on the audit's own counterexample: lam^2 = +1e-20
    # is a positive real, i.e. an EVANESCENT mode, and must keep Re(lam) >= 0
    r = np.array([1e-20 - 1e-30j])
    lam = _sqrt_decay(r)
    assert float(np.real(lam[0])) >= 0.0, lam
    scale = max(float(np.max(np.abs(r))), 1.0)
    assert abs(float(np.real(lam[0]))) <= band * scale
    # ... while a genuinely ON-CUT root (near the IMAGINARY axis) still flips,
    # which is what the predicate exists for
    on_cut = _sqrt_decay(np.array([-1e-20 - 1e-30j]))
    assert abs(float(np.real(on_cut[0]))) <= band * scale

    # (b) the consequence on a real PMM-2D layer spectrum
    from lumenairy.elements.pmm.twod import (
        _build_axis,
        _layer_modes_projected,
        _scalar_projected_ops,
    )
    walls = [_P / 4, 3 * _P / 4]
    ax = _build_axis(_P, walls, 11, [1, 1, 1], False)
    ay = _build_axis(_P, [], 11, [1], False)
    tile = np.array([[1.0 + 0j], [12.25 + 0j], [1.0 + 0j]])
    o5 = np.arange(-5, 6)
    lops = _scalar_projected_ops(ax, ay, tile, o5, o5, _P, _P)
    k0 = 2.0 * np.pi / _WL
    GxF = lops["Gx0F"] / k0
    GyF = lops["Gy0F"] / k0
    _W, _V, lam_l = _layer_modes_projected(
        GxF, GyF, lops["EpsF"], lops["EinvF"], lops["EpnF"],
        formulation="li", EpnxF=lops["EpnxF"], EpnyF=lops["EpnyF"])
    max_lam = float(np.max(np.abs(lam_l)))
    bound = float(np.exp(band * max_lam * k0 * _DEP))
    grow = float(np.max(np.abs(np.exp(-lam_l * k0 * _DEP))))
    assert grow <= bound
    # ... and the bound is tight enough to be a real statement here
    assert bound < 1.0 + 1e-5, (bound, max_lam, k0 * _DEP)
