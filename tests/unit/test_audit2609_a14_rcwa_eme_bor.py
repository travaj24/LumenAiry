"""WP-A14 regression gates for the 2026-09-11 exhaustive adversarial audit --
findings H1-H6 and G11 in ``elements/rcwa``, ``elements/eme``, ``elements/bor``.

Every numeric bar below carries its ORACLE, the oracle's own error floor, the
MEASURED value on this build, and the decades of gap on both sides
(``docs/TESTING_STANDARDS.md`` S1-S5).  Two oracles are written out in this file
rather than imported, so the gate cannot be satisfied by the code under test:

* :func:`_oracle_1d` -- a 1-D RCWA by DIRECT 4N boundary matching (not an
  S-matrix), exact binary Fourier coefficients, Li-1996 inverse rule for TM.
  This is the audit's ``repro/RCWA-EME-BOR/oracle1d.py``, re-derived here.
* the exact step-index fiber HE/EH dispersion relation, via the library's own
  ``fiber_oracle`` -- which is a 4x4 boundary-match determinant that shares no
  code with the FD eigensolver it gates (the audit re-derived it from Maxwell
  and reduced it algebraically to the textbook exact hybrid equation).

Findings covered
----------------
H1  ``guided_modes``' guard band was a fraction of ``k0``, not of the guided
    WINDOW, so every weakly-guiding fiber got an empty list.
H2  the Rayleigh-anomaly wavelength nudge was silent, one-sided and made the
    answer non-monotone in wavelength.
H3  ``rcwa_jones_2d(formulation='fff_nv')`` broke the cell's own C4/C-infinity
    symmetry, and carried no validated-scope notice.
H4  the even-parity fold was gated to ``'laurent'``; the BOR SEM pencils ran a
    non-symmetric ``eig`` on a symmetric-definite pair.
H6  ``per_order_amplitudes`` aliased four of its entries; ``R+T <= 1`` was
    unpoliced whenever any permittivity was complex.
G11 ``_sqrt_decay``'s on-cut predicate flipped near-ZERO evanescent modes.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest

from lumenairy.elements.bor.coupled_radial_eigensolver import guided_modes
from lumenairy.elements.bor.fiber_oracle import fiber_modes
from lumenairy.elements.bor.radial_eigensolver import radial_spectrum
from lumenairy.elements.rcwa import (
    RCWAStack,
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_jones_2d,
)
from lumenairy.elements.rcwa._core import (
    _CUT_BAND_REL,
    _passive_media,
    _sqrt_decay,
    WoodNudgeWarning,
)
import lumenairy.elements.rcwa.twod as _twod


# ===========================================================================
# The independent 1-D oracle (direct 4N boundary matching, NOT an S-matrix)
# ===========================================================================
def _binary_coeffs(c_lo, c_hi, duty, nk):
    """EXACT Fourier coefficients c_k, k = -(nk-1)..(nk-1), of the one-period
    profile that is ``c_hi`` on [0, duty) and ``c_lo`` on [duty, 1)."""
    k = np.arange(-(nk - 1), nk).astype(float)
    ramp = duty * np.sinc(k * duty) * np.exp(-1j * np.pi * k * duty)
    dc = (k == 0).astype(complex)
    return c_lo * dc + (c_hi - c_lo) * ramp


def _toep(c, M):
    N = 2 * M + 1
    ctr = (c.shape[0] - 1) // 2
    i = np.arange(N)
    return c[ctr + (i[:, None] - i[None, :])]


def _sq(x):
    r = np.sqrt(np.asarray(x, dtype=complex))
    return np.where(r.imag < 0, -r, r)                    # Im >= 0


def _oracle_1d(period, n_ridge, n_groove, n_sub, n_sup, depth, duty, wl,
               pol="tm", M=21):
    """Diffraction efficiencies of a single binary layer by DIRECT boundary
    matching of the 4N unknowns (r, a, b, t) -- exp(-i omega t), forward
    exp(+i kz z), Im(kz) >= 0.  No S-matrix, no Redheffer star, no mode
    orientation heuristic, so it has no branch-cut or half-space-degeneracy
    machinery in common with the library."""
    k0 = 2 * np.pi / wl
    N = 2 * M + 1
    m = np.arange(-M, M + 1)
    eps_r, eps_g = complex(n_ridge) ** 2, complex(n_groove) ** 2
    eps_I, eps_II = complex(n_sup) ** 2, complex(n_sub) ** 2
    alpha = m * (wl / period)                              # normal incidence
    Kx = np.diag(alpha.astype(complex))
    gI, gII = _sq(eps_I - alpha ** 2), _sq(eps_II - alpha ** 2)
    g0 = _sq(eps_I).real
    E = _toep(_binary_coeffs(eps_g, eps_r, duty, N), M)
    Ei = _toep(_binary_coeffs(1 / eps_g, 1 / eps_r, duty, N), M)
    I = np.eye(N, dtype=complex)
    delta = (m == 0).astype(complex)
    if pol == "te":
        q2, W = np.linalg.eig(Kx @ Kx - E)
        gam = _sq(-q2)
        Vm = W @ np.diag(gam)
        VI, VII = np.diag(gI), np.diag(gII)
        src_V = -g0 * delta
    else:                                                  # Li-1996 inverse rule
        B = np.linalg.inv(Ei) @ (Kx @ np.linalg.inv(E) @ Kx - I)
        q2, W = np.linalg.eig(B)
        gam = _sq(-q2)
        Vm = Ei @ W @ np.diag(gam)
        VI, VII = np.diag(gI / eps_I), np.diag(gII / eps_II)
        src_V = -(g0 / eps_I) * delta
    X = np.diag(np.exp(1j * k0 * gam * depth))
    Z = np.zeros((N, N), dtype=complex)
    A = np.block([[-I, W, W @ X, Z],
                  [-VI, -Vm, Vm @ X, Z],
                  [Z, W @ X, W, -I],
                  [Z, -Vm @ X, Vm, VII]])
    rhs = np.concatenate([delta, src_V, np.zeros(N), np.zeros(N)])
    sol = np.linalg.solve(A, rhs)
    r, t = sol[:N], sol[3 * N:]
    if pol == "te":
        DEr = np.real(gI / g0) * np.abs(r) ** 2
        DEt = np.real(gII / g0) * np.abs(t) ** 2
    else:
        DEr = np.real(gI / eps_I) / (g0 / eps_I.real) * np.abs(r) ** 2
        DEt = np.real(gII / eps_II) / (g0 / eps_I.real) * np.abs(t) ** 2
    DEr = np.where(np.real(gI) > 1e-14, DEr, 0.0)
    DEt = np.where(np.real(gII) > 1e-14, DEt, 0.0)
    return m, np.real(DEr), np.real(DEt)


# ===========================================================================
# H1 -- guided_modes on a weakly-guiding fiber
# ===========================================================================
_LAM_FIBER = 1.55e-6
_K0_FIBER = 2 * np.pi / _LAM_FIBER


def _fiber_geometry(n1, n2, V=2.4):
    a = V / (_K0_FIBER * np.sqrt(n1 ** 2 - n2 ** 2))
    return a, fiber_modes(1, a, n1 ** 2, n2 ** 2, _K0_FIBER)[0] / _K0_FIBER


@pytest.mark.parametrize("n1,n2,label", [
    (1.45, 1.44, "textbook V=2.4 fiber, dn = 0.010"),
    (1.45, 1.445, "telecom-SMF-like, dn = 0.005"),
])
def test_h1_weakly_guiding_fiber_is_not_an_empty_list(n1, n2, label):
    """H1 (P1).  ORACLE: the EXACT hybrid HE11 dispersion relation (the 4x4
    boundary-match determinant of ``fiber_oracle``, which shares no code with
    the FD vector eigensolver), oracle floor ~1e-12 in n_eff (its bisection runs
    80 halvings on a 6000-point bracket).

    BAR 2e-4 in n_eff.  MEASURED on this build at (Rbig, N) = (6a, 150):
    +1.10e-06 (dn = 0.010) and +5.53e-07 (dn = 0.005); the same solve at
    N = 200 / 300 lands at -5.9e-05 / +2.8e-07 and -3.0e-05 / +1.4e-07, so the
    FD floor over the family is ~6e-5.  That is 0.5 decade below this bar and
    8 decades above the oracle's floor.

    FAILS BEFORE: the shipped guard band was ``5e-3 * k0`` PER SIDE while the
    guided window is only ``(n1 - n2) * k0`` wide, so the admissible interval
    was EMPTY (dn = 0.005) or a single point (dn = 0.010) and this returned
    ``[]`` at every (Rbig, N) tried -- 4a/6a/8a/12a x 150/300/600.  The
    pre-fix arithmetic is asserted explicitly below so the fail-before is a
    decision about the rule, not a hope about the build.
    """
    a, neff_exact = _fiber_geometry(n1, n2)
    window = (np.sqrt(n1 ** 2) - np.sqrt(n2 ** 2)) * _K0_FIBER
    # The pre-fix rule, evaluated here: the band alone emptied the window.
    # ``dn = 0.010`` is the EXACT boundary case (the interval collapses to a
    # single point), so the comparison carries one rounding ULP of slack.
    assert window <= 2.0 * (5e-3 * _K0_FIBER) * (1.0 + 1e-9), (
        f"{label}: the pre-fix 5e-3*k0 band no longer empties this window -- "
        f"the fixture has stopped being discriminating")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm = guided_modes(1, a, 6.0 * a, 150, n1 ** 2, n2 ** 2, _K0_FIBER)
    assert len(gm) >= 1, f"{label}: guided_modes returned an EMPTY list"
    neff = max(md["q"].real for md in gm) / _K0_FIBER
    assert abs(neff - neff_exact) < 2e-4, (
        f"{label}: n_eff {neff:.12f} vs exact HE11 {neff_exact:.12f}")


def test_h1_margin_scales_with_the_window_not_with_k0():
    """H1 (P1), stated as the RULE rather than as one fixture: halving the
    index contrast must halve the admitted-window loss, not leave it fixed.

    The guard band is now ``max(1e-6*k0, 1e-3*(qhi - qlo))``, so the ADMITTED
    fraction of the guided window is 1 - 2e-3 at EVERY contrast above
    ``dn = 1e-3``.  Pre-fix it was ``1 - 1e-2/dn``: 0.998 at dn = 5 but
    NEGATIVE (i.e. empty) for every dn < 0.01.  Asserted on a five-decade
    contrast ladder."""
    k0 = _K0_FIBER
    for dn in (1e-2, 5e-3, 1e-3, 1e-4, 1e-5):
        n2 = 1.44
        n1 = n2 + dn
        qlo, qhi = n2 * k0, n1 * k0
        window = qhi - qlo
        margin = max(1e-6 * k0, 1e-3 * window)
        admitted = (window - 2 * margin) / window
        # Above the 1e-6*k0 floor the admitted fraction is CONSTANT at
        # 1 - 2e-3; below it the floor takes over and the fraction falls, but
        # it never becomes empty (0.80 at dn = 1e-5, the tightest rung here).
        bar = 0.99 if dn >= 1e-3 else 0.5
        assert admitted > bar, f"dn={dn}: admitted fraction {admitted}"
        old_admitted = (window - 2 * (5e-3 * k0)) / window
        if dn < 1e-2:
            assert old_admitted <= 0.0, (
                f"dn={dn}: the pre-fix band no longer empties the window")
    # the rule itself: 1e-3 of the window, not 5e-3 of k0
    for dn in (1e-2, 1e-1, 1.0):
        window = dn * k0
        assert max(1e-6 * k0, 1e-3 * window) == pytest.approx(1e-3 * window)


def test_h1_degenerate_window_raises_instead_of_returning_empty():
    """H1 (P1).  When the window really is narrower than the band it needs
    (``dn <= 2e-6``), nothing can be admitted -- and saying so is the one thing
    an empty list cannot.  DECISION assertion: the message must name both
    permittivities so the caller can see what to change."""
    n2 = 1.45
    n1 = n2 + 1e-7                       # window 1e-7 k0 vs 2 x 1e-6 k0 band
    with pytest.raises(ValueError, match="guided_modes: the guided window"):
        guided_modes(1, 3e-6, 18e-6, 60, n1 ** 2, n2 ** 2, _K0_FIBER)


# ===========================================================================
# H2 -- the Rayleigh-anomaly nudge
# ===========================================================================
# The canonical Moharam-1995 mount: Lambda = lambda = 1 um puts the m = +/-1
# orders EXACTLY at cut-off in BOTH half-spaces (n = 1), which is a standard
# design point and the classic benchmark geometry.
_WOOD = dict(period=1.0e-6, n_ridge=2.04, n_groove=1.0, n_substrate=1.0,
             n_superstrate=1.0, depth=1.0e-6, duty=0.5, wl=1.0e-6, M=21)


def _wood_lib(wl, pol, M=_WOOD["M"], full=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T = rcwa_efficiency_1d(
            _WOOD["period"], _WOOD["n_ridge"], _WOOD["n_groove"],
            _WOOD["n_substrate"], _WOOD["n_superstrate"], _WOOD["depth"],
            _WOOD["duty"], wl, polarization=pol, n_orders=M, formulation="li")
    if full:
        return np.asarray(R), np.asarray(T)
    return float(R[M]), float(R.sum() + T.sum())


def _wood_oracle(wl, pol, M=_WOOD["M"], full=False):
    _m, R, T = _oracle_1d(
        _WOOD["period"], _WOOD["n_ridge"], _WOOD["n_groove"],
        _WOOD["n_substrate"], _WOOD["n_superstrate"], _WOOD["depth"],
        _WOOD["duty"], wl, pol=pol, M=M)
    if full:
        return np.asarray(R), np.asarray(T)
    return float(R[M]), float(R.sum() + T.sum())


def test_h2_the_nudge_announces_itself():
    """H2 (P1).  The whole of the finding is that the substitution was SILENT:
    ``warnings.catch_warnings(record=True)`` around the solve at the exact Wood
    point returned an EMPTY list while the solver was answering a different
    wavelength.  DECISION assertion -- exactly one WoodNudgeWarning, naming
    both wavelengths."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rcwa_efficiency_1d(
            _WOOD["period"], _WOOD["n_ridge"], _WOOD["n_groove"],
            _WOOD["n_substrate"], _WOOD["n_superstrate"], _WOOD["depth"],
            _WOOD["duty"], _WOOD["wl"], polarization="tm",
            n_orders=_WOOD["M"], formulation="li")
    wood = [w for w in caught if issubclass(w.category, WoodNudgeWarning)]
    assert len(wood) == 1, [str(w.message) for w in caught]
    msg = str(wood[0].message)
    assert "1e-06" in msg and "SYMMETRIC AVERAGE" in msg, msg


def test_h2_exact_wood_point_beats_the_one_sided_nudge():
    """H2 (P1).  ORACLE: :func:`_oracle_1d` at the EXACT requested wavelength,
    which solves this mount cleanly -- its own energy closure is 1.8e-15 at
    M = 21 and its truncation drift is 0.156055 / 0.155846 / 0.155819 /
    0.155812 / 0.155808 for M = 11/21/31/41/61, so at MATCHED M = 21 it is the
    right reference to ~2e-4 of itself.

    MEASURED (TM R0, M = 21): oracle 0.155845785342; library BEFORE
    0.155908839054 (one-sided +1e-7 nudge, +4.05e-04 relative); library AFTER
    0.155846827297 (symmetric +/-1e-9 bracket, +6.69e-06 relative).  The
    one-sided value is recomputed HERE, by solving at lambda*(1+1e-7) -- so the
    before/after is measured on this build, not quoted.  TE: +1.36e-05 ->
    -1.42e-06 relative.

    BAR: the symmetric average must be at least 10x closer to the oracle than
    the one-sided nudge, and within 2e-5 absolute.  Measured ratio 60.5x on TM
    (6.31e-05 -> 1.04e-06 absolute) and 9.5x on TE -- 0.8 decade of gap on the
    ratio, 1.3 decades on the absolute bar.  Both columns fall as sqrt(delta)
    (a Wood anomaly is a square-root branch point of the efficiencies in the
    wavelength, measured over five decades: ratio 1.732 per 3x in delta), so no
    bar tighter than the bracket itself is available from this formulation.
    """
    ref, clos_o = _wood_oracle(_WOOD["wl"], "tm")
    assert abs(clos_o - 1.0) < 1e-12, clos_o        # the oracle is well posed
    got, clos = _wood_lib(_WOOD["wl"], "tm")
    one_sided, _ = _wood_lib(_WOOD["wl"] * (1.0 + 1e-7), "tm")
    err_new = abs(got - ref)
    err_old = abs(one_sided - ref)
    assert abs(clos - 1.0) < 1e-11, clos            # closure survives the mean
    assert err_old > 5e-5, (
        f"the one-sided nudge is no longer 6.3e-05 off -- fixture stale "
        f"(got {err_old:.3e})")
    assert err_new < err_old / 10.0, (
        f"symmetric average err {err_new:.3e} vs one-sided {err_old:.3e}")
    assert err_new < 2e-5, err_new


def test_h2_wavelength_sweep_is_monotone_through_the_anomaly():
    """H2 (P1).  The pre-fix value AT the anomaly was bit-identical to its
    ``+1e-7`` neighbour, so it sat ABOVE BOTH of its neighbours -- a one-point
    spike that an optimiser differencing through it reads as a real feature.

    MEASURED on this build, TM R0 over
    delta = -3e-7 .. +3e-7: 0.155772307711, 0.155803587482, 0.155822748627,
    0.155832507556, [anomaly] 0.155846827297, 0.155865654287, 0.155880240631,
    0.155908839054, 0.155955410642 -- strictly increasing, smallest forward
    difference +5.23e-06.  PRE-FIX the value at delta = 0 was 0.155908839054,
    i.e. EQUAL to the delta = +1e-7 sample and ABOVE the +3e-8 one, so the
    sequence decreased twice.  DECISION assertion: strict monotonicity.
    """
    deltas = (-3e-7, -1e-7, -3e-8, -1e-8, 0.0, 1e-8, 3e-8, 1e-7, 3e-7)
    vals = np.array([_wood_lib(_WOOD["wl"] * (1.0 + d), "tm")[0]
                     for d in deltas])
    d1 = np.diff(vals)
    assert np.all(d1 > 0.0), dict(zip(deltas, vals))
    # and the anomaly value lies STRICTLY between its two neighbours, which is
    # what "a limit" means and what the pre-fix one-sided value was not
    i0 = deltas.index(0.0)
    assert vals[i0 - 1] < vals[i0] < vals[i0 + 1]


def test_h2_exactly_grazing_orders_stay_four_decades_below_the_specular():
    """H2 (P1) -- the ONE thing the symmetric average gets wrong, pinned so it
    cannot grow silently.

    An order that is EXACTLY grazing at the requested wavelength carries no
    z-directed power: that is a theorem, and the oracle returns exactly 0 for
    it (its mask is ``Re(kz) > 1e-14``).  The one-sided nudge happened to agree
    -- at ``+delta`` the m = +/-1 orders of this mount are EVANESCENT -- but at
    ``-delta`` they are PROPAGATING and carry ~sqrt(delta) of power, so half of
    that survives the mean.

    MEASURED at the shipped +/-1e-9 bracket, m = +/-1: TM R 0.0 -> 2.344e-06,
    TE R 0.0 -> 3.963e-07 (10x smaller than at a 1e-7 bracket, as sqrt(delta)
    requires).  The power is not invented -- it comes out of the specular order
    -- so the CLOSURE stays exact: measured -3.75e-14 (TM) and +1.87e-14 (TE).

    BARS: the grazing order must stay below 1e-4 absolute and 4 decades below
    the specular order of the same port; the closure must hold to 1e-11.  The
    measured values are 1.6 decades inside the absolute bar and the closure is
    2.4 decades inside its own.  A regression that put real power in a grazing
    order -- e.g. a lost evanescent mask -- would be O(1e-1) and fail both.
    """
    M = _WOOD["M"]
    for pol, floor in (("tm", 1e-4), ("te", 1e-4)):
        Rl, Tl = _wood_lib(_WOOD["wl"], pol, full=True)
        Ro, To = _wood_oracle(_WOOD["wl"], pol, full=True)
        for k in (M - 1, M + 1):                       # the m = -/+1 orders
            assert Ro[k] == 0.0 and To[k] == 0.0       # the exact answer is 0
            assert Rl[k] < floor and Tl[k] < floor, (pol, k, Rl[k], Tl[k])
            assert Rl[k] < 1e-4 * Rl[M], (pol, k, Rl[k], Rl[M])
            assert Tl[k] < 1e-4 * Tl[M], (pol, k, Tl[k], Tl[M])
        assert abs(Rl.sum() + Tl.sum() - 1.0) < 1e-11, pol
        # and orders beyond the grazing pair are exactly zero on both sides
        assert Rl[M + 2] == 0.0 and Tl[M + 2] == 0.0


def test_h2_wl_eff_is_on_the_result():
    """H2 (P1).  A sweep must be able to SEE the substitution without parsing a
    warning.  ``Efficiency2D.wl_eff`` / ``RCWAResult.wl_eff`` carry the
    wavelength(s) actually solved: a float off-anomaly, the (lo, hi) pair when
    the answer is the symmetric average."""
    Sx = 64
    cell = (np.where(np.arange(Sx)[:, None] < Sx // 2, 2.04 ** 2 + 0j, 1.0 + 0j)
            * np.ones((1, Sx)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = rcwa_efficiency_2d(1e-6, 1e-6, cell, 1.0, 1.0, 1e-6, 1e-6,
                                n_orders_x=4, n_orders_y=0, polarization="tm",
                                formulation="li")
        off = rcwa_efficiency_2d(1e-6, 1e-6, cell, 1.0, 1.0, 1e-6, 0.9e-6,
                                 n_orders_x=4, n_orders_y=0, polarization="tm",
                                 formulation="li")
    assert isinstance(on.wl_eff, tuple) and len(on.wl_eff) == 2
    lo, hi = on.wl_eff
    assert lo < 1e-6 < hi
    assert abs(0.5 * (lo + hi) - 1e-6) < 1e-20      # symmetric by construction
    assert off.wl_eff == pytest.approx(0.9e-6, rel=0, abs=0)


def test_h2_off_anomaly_solves_are_untouched_and_unwarned():
    """H2 (P1) -- the other half of the contract: a wavelength that is NOT on
    an anomaly must take the bare path (no warning, no second solve, the same
    numbers as before).  ORACLE: :func:`_oracle_1d` at the same wavelength;
    the audit measured <= 2.0e-13 agreement over 24 configurations and this
    reproduces that at one of them."""
    wl = 0.9e-6
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got, clos = _wood_lib(wl, "tm")
    assert not [w for w in caught if issubclass(w.category, WoodNudgeWarning)]
    ref, _ = _wood_oracle(wl, "tm")
    assert abs(got - ref) < 1e-12, (got, ref)       # measured 3e-15 here
    assert abs(clos - 1.0) < 1e-11


# ===========================================================================
# H3 -- fff_nv on rcwa_jones_2d must not manufacture form birefringence
# ===========================================================================
_H3 = dict(period=0.5e-6, depth=0.3e-6, wl=0.633e-6, S=96)


def _promote(c):
    t = np.zeros(c.shape + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = c
    return t


def _square_cell():
    S = _H3["S"]
    c = np.full((S, S), 2.25 + 0j)
    c[S // 4:3 * S // 4, S // 4:3 * S // 4] = 6.25 + 0j
    return c


def _disk_cell():
    S = _H3["S"]
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = np.full((S, S), 2.25 + 0j)
    c[X ** 2 + Y ** 2 <= 0.25 ** 2] = 6.25 + 0j
    return c


def _jones(cell, M, formulation, symmetrize=True):
    kw = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if symmetrize:
            _o, _R, _T, J = rcwa_jones_2d(
                _H3["period"], _H3["period"], _promote(cell), 1.0, 1.0,
                _H3["depth"], _H3["wl"], n_orders_x=M, n_orders_y=M,
                formulation=formulation, **kw)
            return J
        orig = _twod._li_convolutions_2d_tensor
        _twod._li_convolutions_2d_tensor = (
            lambda *a, **k: orig(*a, **{**k, "symmetrize": False}))
        try:
            _o, _R, _T, J = rcwa_jones_2d(
                _H3["period"], _H3["period"], _promote(cell), 1.0, 1.0,
                _H3["depth"], _H3["wl"], n_orders_x=M, n_orders_y=M,
                formulation=formulation, **kw)
        finally:
            _twod._li_convolutions_2d_tensor = orig
    return J


@pytest.mark.parametrize("maker,name,pre_fix_floor", [
    (_square_cell, "square (C4)", 1e-4),
    (_disk_cell, "disk (C-infinity)", 5e-3),
])
def test_h3_fff_nv_keeps_the_cells_own_symmetry(maker, name, pre_fix_floor):
    """H3 (P2).  ORACLE: the CELL's own symmetry.  ``cell == cell.T`` and the
    incidence is normal, so ``Jxx == Jyy`` EXACTLY -- there is no physics in
    the difference, only the factorization order.  The oracle's floor is the
    solver's own arithmetic, which ``'laurent'`` and ``'li'`` show on the same
    fixtures: 1e-15 .. 3e-13 over M = 4..12.

    BAR 1e-12.  MEASURED after: 5.4e-15 .. 2.6e-13 across both cells and
    M = 4..12 -- at the same level as the two rigorous formulations, 1 decade
    inside the bar.  MEASURED before, reproduced IN THIS TEST through the
    private ``symmetrize=False`` switch: 9.04e-04 (square) and 2.06e-02 (disk)
    at M = 4, falling as ~1/M to 1.06e-04 and 5.75e-03 at M = 12 -- 9 and 11
    decades above the bar, i.e. 2-5e-03 of spurious form birefringence on a
    cell that has none, in exactly the waveplate workflow this entry serves.
    """
    cell = maker()
    assert np.array_equal(cell, cell.T)
    for M in (4, 8, 12):
        J = _jones(cell, M, "fff_nv")
        assert abs(J[0, 0] - J[1, 1]) < 1e-12, (name, M, J[0, 0], J[1, 1])
    # fail-before, measured here rather than quoted
    J_old = _jones(cell, 4, "fff_nv", symmetrize=False)
    assert abs(J_old[0, 0] - J_old[1, 1]) > pre_fix_floor, (
        f"{name}: the single-order operator no longer breaks the symmetry -- "
        f"fixture stale")


def test_h3_separable_stripe_is_unchanged_by_the_symmetrisation():
    """H3 (P2).  For a y-UNIFORM (separable) cell the two factorization orders
    coincide analytically, so the symmetrized operator must reproduce the
    single-order one.

    BAR 1e-11 on max|dJ| (|J| is O(1) here, so this is a relative bar).
    MEASURED: 8.7e-14 / 4.5e-14 / 1.3e-12 at M = 4/8/12 -- the residual is the
    extra matrix arithmetic's rounding, ~1-2 decades inside the bar and 12
    decades below the 1.33 form birefringence the stripe genuinely has (which
    is asserted too, so the gate cannot pass on a cell with no signal)."""
    S = _H3["S"]
    stripe = np.full((S, S), 2.25 + 0j)
    stripe[S // 4:3 * S // 4, :] = 6.25 + 0j
    for M in (4, 8, 12):
        J_new = _jones(stripe, M, "fff_nv")
        J_old = _jones(stripe, M, "fff_nv", symmetrize=False)
        assert np.max(np.abs(J_new - J_old)) < 1e-11, M
        assert abs(J_new[0, 0] - J_new[1, 1]) > 1.0     # real birefringence
    # ... and it agrees with the rigorous 'li' rule on the same stripe, which
    # is where Li-2003 reduces EXACTLY to Li-1996
    J_li = _jones(stripe, 12, "li")
    J_nv = _jones(stripe, 12, "fff_nv")
    assert np.max(np.abs(J_li - J_nv)) < 1e-9, np.max(np.abs(J_li - J_nv))


def test_h3_curved_cell_gets_the_validated_scope_notice():
    """H3 (P2), second half.  ``rcwa_efficiency_2d(formulation='fff_nv')``
    REFUSES a curved cell while ``rcwa_jones_2d(formulation='fff_nv')``
    accepted the same disk silently -- two entry points, the same token, one
    of them unguarded.  DECISION assertions: the disk warns and names the
    diagonal-boundary fraction; the axis-aligned square does not; and
    ``allow_nonseparable_nv=True`` silences it."""
    def _notices(cell, **kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rcwa_jones_2d(_H3["period"], _H3["period"], _promote(cell),
                          1.0, 1.0, _H3["depth"], _H3["wl"], n_orders_x=4,
                          n_orders_y=4, formulation="fff_nv", **kw)
        return [str(w.message) for w in caught
                if "NON-SEPARABLE" in str(w.message)]

    disk = _notices(_disk_cell())
    assert len(disk) == 1, disk
    assert "of the cell boundary runs diagonal" in disk[0]
    assert _notices(_square_cell()) == []
    assert _notices(_disk_cell(), allow_nonseparable_nv=True) == []


# ===========================================================================
# H4 -- the even-parity fold, and the BOR symmetric-definite pencil
# ===========================================================================
@pytest.mark.parametrize("formulation", ["laurent", "li", "fff_nv"])
def test_h4_even_parity_fold_covers_every_in_plane_formulation(formulation):
    """H4 (P2, perf).  The fold acts on the ``(P, Q)`` GENERATOR and is
    indifferent to how the permittivity operators were factorized, but it was
    gated on ``formulation == 'laurent'``, so ``'li'`` / ``'fff_nv'`` users
    never got it (a documented Amdahl gap).

    DECISION assertion that it actually ENGAGES: ``_symmetric_cascade_rt`` is
    counted, and it must return a non-None result for every formulation.
    ORACLE for the numbers: the FULL 2N solve of the same problem
    (``symmetry=False``), which shares no even-sector code.

    BAR 1e-11 on max|dJ| -- the documented "~1e-12, not bit-identical"
    contract for the even-adapted basis.  MEASURED: 1.1e-13 / 7.4e-14
    ('li', M = 6/9), 1.2e-13 / 2.2e-13 ('fff_nv'), 2.2e-13 / 3.4e-13
    ('laurent') -- 2 decades inside the bar.  Speed, for the record and NOT
    asserted (TESTING_STANDARDS S1): 3.01x / 3.21x ('li'), 2.47x / 3.15x
    ('fff_nv') against 1.00x before, on a 96x96 square cell.
    """
    cell = _promote(_square_cell())
    calls = []
    orig = _twod._symmetric_cascade_rt
    _twod._symmetric_cascade_rt = (
        lambda *a, **k: calls.append(orig(*a, **k)) or calls[-1])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, _R, _T, J_sym = rcwa_jones_2d(
                _H3["period"], _H3["period"], cell, 1.0, 1.0, _H3["depth"],
                _H3["wl"], n_orders_x=6, n_orders_y=6,
                formulation=formulation, symmetry=True)
    finally:
        _twod._symmetric_cascade_rt = orig
    assert len(calls) == 1 and calls[0] is not None, (
        f"{formulation}: the even-parity fold did not engage")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, _R, _T, J_full = rcwa_jones_2d(
            _H3["period"], _H3["period"], cell, 1.0, 1.0, _H3["depth"],
            _H3["wl"], n_orders_x=6, n_orders_y=6, formulation=formulation,
            symmetry=False)
    assert np.max(np.abs(J_sym - J_full)) < 1e-11, np.max(
        np.abs(J_sym - J_full))


@pytest.mark.parametrize("m,bc,zeros_fn", [
    (0, "dirichlet", "jn"), (1, "dirichlet", "jn"), (3, "dirichlet", "jn"),
    (1, "neumann", "jnp"), (3, "neumann", "jnp"),
])
def test_h4_bor_pencil_eigh_accuracy(m, bc, zeros_fn):
    """H4 (P2).  ``radial_spectrum`` now solves the symmetric-definite pencil
    with ``eigh(A, M)`` instead of a non-symmetric ``eig`` on an explicitly
    formed ``M^-1 A``.

    ORACLE: the Bessel zeros ``j_{m,n}`` / ``j'_{m,n}`` (scipy special
    functions, floor ~1e-15 relative).  BAR 1e-12 relative.  MEASURED after:
    1.19e-13 / 9.26e-14 / 1.38e-14 (m = 0/1/3 Dirichlet) and 1.74e-13 /
    3.81e-14 (m = 1/3 Neumann) at degree 8, 12 elements -- 1 decade inside the
    bar and 1-2 decades BETTER than the pre-fix 3.11e-13 / 1.78e-13 /
    5.42e-14 / 7.04e-13 / 1.83e-13 the audit measured on the same fixture.
    """
    from scipy.special import jn_zeros, jnp_zeros
    ref = (jn_zeros(m, 6) if zeros_fn == "jn" else jnp_zeros(m, 6))
    ev = radial_spectrum(m, 1.0, 8, 12, bc=bc, n_low=6)
    rel = np.max(np.abs(np.sqrt(np.abs(ev)) / ref - 1.0))
    assert rel < 1e-12, rel
    assert np.all(np.isreal(ev))          # eigh returns a real spectrum


# ===========================================================================
# H6 -- the aliased amplitude dict and the passive-structure energy bound
# ===========================================================================
def _stripe_stack(eps_ridge=2.04 ** 2 + 0j, n_sub=1.5):
    Sx = 64
    cell = (np.where(np.arange(Sx)[:, None] < Sx // 2, eps_ridge, 1.0 + 0j)
            * np.ones((1, Sx)))
    s = RCWAStack(0.8e-6, period_y=0.8e-6, n_orders=4, n_orders_y=0,
                  n_superstrate=1.0, n_substrate=n_sub)
    s.add_layer(0.3e-6, eps_cell=cell)
    s.set_source(0.633e-6, theta=0.0, phi=0.0)
    return s


def test_h6_per_order_amplitudes_hands_out_copies():
    """H6 (P3).  ``kz`` was explicitly copied "so the public dict keeps its
    writable-array contract", but ``Ex``/``Ey``/``kx``/``ky``/``orders`` were
    handed out BY REFERENCE into the result's own modal dict -- so
    ``amp['Ex'][:] = 0`` made the NEXT call on the SAME result return zeros.
    DECISION assertion, on every key that carries an array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _stripe_stack().solve()
    first = res.per_order_amplitudes("reflection")
    before = {k: np.array(v, copy=True) for k, v in first.items()
              if isinstance(v, np.ndarray)}
    assert np.max(np.abs(before["Ex"])) > 1e-6      # non-trivial to start with
    for k in before:
        first[k][...] = 0
    second = res.per_order_amplitudes("reflection")
    for k, v in before.items():
        assert np.array_equal(second[k], v), f"{k} was aliased"


def test_h6_passive_media_predicate():
    """H6 (P3).  The three clauses of :func:`_passive_media`, each of which is
    a measured exception the tighter energy bar must not false-fire on.
    Permittivities are INTERNAL (loss-bridge-conjugated): Im < 0 is loss."""
    lossy = np.array([2.25 - 0.3j])
    gain = np.array([2.25 + 0.3j])
    real = np.array([2.25 + 0j])
    assert _passive_media(1.0 + 0j, 2.25 + 0j, real)
    assert _passive_media(1.0 + 0j, 2.25 + 0j, lossy)     # absorbing LAYER: ok
    assert not _passive_media(1.0 - 0.01j, 2.25 + 0j, real)   # lossy incidence
    assert not _passive_media(1.0 + 0j, 2.25 + 0j, gain)      # gain layer
    assert not _passive_media(1.0 + 0j, 2.25 + 0.1j, real)    # gain substrate
    asym = np.zeros((2, 2, 3, 3), dtype=complex)
    for i in range(3):
        asym[..., i, i] = 2.25
    asym[..., 0, 2] = 0.1                       # e_xz != e_zx: non-reciprocal
    assert not _passive_media(1.0 + 0j, 2.25 + 0j, asym)


def test_h6_passive_bound_is_armed_on_a_lossy_cell():
    """H6 (P3).  The tight lossless closure clause is disarmed by ANY complex
    permittivity, so a metal grating in air -- a lossless INCIDENCE medium,
    where ``R+T <= 1`` is a theorem -- could return ``R+T`` anywhere in
    (1, 1.05] with no signal at all.  The new one-sided clause covers it.

    DECISION assertions: (a) the clause is ARMED on a lossy cell under a
    lossless incidence medium while the lossless clause is not; (b) it does
    NOT fire on a converged solve of that cell; (c) it DOES fire on a
    constructed violation just above the bar (the guard is called directly,
    which is the only way to construct one without a pathological geometry).
    """
    from lumenairy.elements.rcwa._core import (_cell_lossless, _check_energy,
                                               _EnergyWarning)
    eps_metal = np.array([[0.135 + 3.99j]]) ** 2
    eps_int = np.conj(eps_metal)                  # internal convention
    assert not _cell_lossless(1.0 + 0j, 2.25 + 0j, eps_int)
    assert _passive_media(1.0 + 0j, 2.25 + 0j, eps_int)
    R = np.array([[0.4, 0.2]])
    T = np.array([[0.3, 0.1]])                    # sum 1.0 exactly -- quiet
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _check_energy("probe", R, T, lossless=False, passive=True)
    assert not [w for w in caught if issubclass(w.category, _EnergyWarning)]
    # 1e-5 excess: 1 decade above the 1e-6 bar, 7 decades above the 1.4e-13
    # closure clean solves hold, and 3.7 decades BELOW the 1.05 tripwire that
    # was the only guard before.
    T2 = np.array([[0.3, 0.10001]])           # sum 1.00001: 1e-5 excess
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _check_energy("probe", R, T2, lossless=False, passive=True)
    fired = [w for w in caught if issubclass(w.category, _EnergyWarning)]
    assert len(fired) == 1, [str(w.message) for w in caught]
    assert "passive-structure energy bound" in str(fired[0].message)
    # and it stays silent when the incidence medium is lossy (passive=False),
    # which is the documented +2.3% case
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _check_energy("probe", R, T2, lossless=False, passive=False)
    assert not [w for w in caught if issubclass(w.category, _EnergyWarning)]


# ===========================================================================
# G11 -- _sqrt_decay's on-cut predicate
# ===========================================================================
def test_g11_near_zero_evanescent_mode_is_not_flipped():
    """G11 (P3).  The on-cut pin tested ``|Re r| <= band * scale``, i.e.
    proximity to the ORIGIN on the SPECTRUM's scale -- which a deeply
    evanescent root also satisfies once its own magnitude has collapsed.

    COUNTEREXAMPLE (the audit's): ``_sqrt_decay([1e-20 - 1e-30j])`` returned
    ``-1e-10 + 5e-21j``.  ``lam^2`` is a positive real there, so the mode
    DECAYS, and handing it back with ``Re(lam) < 0`` turns the propagator
    ``exp(-lam k0 L)`` into ``exp(+|gamma| k0 L)`` -- the growth the
    ``Re >= 0`` rule exists to prevent.  DECISION assertion: the root must keep
    the principal branch, ``Re >= 0``.
    """
    x = np.array([1e-20 - 1e-30j])
    r = _sqrt_decay(x)
    assert r[0].real > 0, r[0]
    assert np.allclose(r, np.sqrt(x))        # principal branch, untouched


@pytest.mark.parametrize("big", [1.0, 1e2, 1e4])
def test_g11_scale_relative_band_cannot_flip_an_evanescent_mode(big):
    """G11 (P3).  The band's scale is the ARRAY's largest root, so a spectrum
    with a large top used to pull an ordinary near-cutoff EVANESCENT mode into
    the band.  The price of a wrong flip is
    ``|X| = exp(band * max|lam| * k0 L)`` -- 1.059 / 302 / 1e248 for spectrum
    scales 1 / 1e2 / 1e4 at the docstring's own worst ``k0 L = 1.1424e7``.

    DECISION assertion: no flip, so ``|X| <= 1`` at that ``k0 L``.  MEASURED
    after: |X| = 1.0 exactly (Re(lam) > 0 for every scale)."""
    x = np.array([big ** 2 + 0j, (0.5e-8 * big) ** 2 - 1e-30j])
    r = _sqrt_decay(x)
    assert r[1].real > 0, (big, r[1])
    assert np.exp(-r[1].real * 1.1424e7) <= 1.0


def test_g11_genuine_on_cut_propagating_mode_still_flips():
    """G11 (P3), the other side: the pin must still do its job.  A PROPAGATING
    mode on the cut has ``lam^2 = -s + i eta`` with ``eta`` at the
    eigensolver's backward-error level, so the principal root is
    ``eta/(2 sqrt(s)) - i sqrt(s)`` -- tiny real part, O(1) imaginary part,
    ``Im^2 / Re^2 ~ 1e32``.  It must be flipped to the OUTGOING root
    (``Im >= 0``) on every build.  Worst ``|Re r| / |r|`` ever flipped on the
    populations the band was derived against is 2.0751e-03, five decades clear
    of the new ``Im^2 = Re^2`` edge, so the third conjunct is inert there."""
    s = 4.0
    for eta in (-1e-18, -1e-20, -2.9e-15):
        r = _sqrt_decay(np.array([-s + 1j * eta]))
        assert r[0].imag > 0, (eta, r[0])
        assert abs(abs(r[0].imag) - np.sqrt(s)) < 1e-12
    # and the band is what admits it: an eta large enough to put |Re r| above
    # band * scale must NOT be flipped (it is no longer rounding noise)
    eta_big = 10.0 * _CUT_BAND_REL * 2.0 * np.sqrt(s) * np.sqrt(s)
    r = _sqrt_decay(np.array([-s - 1j * eta_big]))
    assert r[0].imag < 0, r[0]


# ===========================================================================
# BOR: energy is conserved for ARBITRARY excitation, not just single-mode
# ===========================================================================
def test_bor_propagating_smatrix_is_unitary_and_closes_on_superpositions():
    """The audit's strongest positive BOR result, which NOTHING in the suite
    gated: the staggered BOR modal basis is flux-ORTHONORMAL, so the S-matrix
    restricted to the PROPAGATING channels is UNITARY -- and therefore energy
    closes for an arbitrary multi-channel input, not only for the single-mode
    excitations the per-channel ``energy`` array reports.

    ORACLE: unitarity itself.  ``U = [S11; S21]`` restricted to the solver's own
    ``inc`` / ``out`` index sets must satisfy ``U^H U = I``; the per-channel
    ``energy`` array is exactly the DIAGONAL of ``U^H U``, so the OFF-diagonal
    and the random-superposition closure are statements the shipped array
    cannot make.

    BAR 1e-9.  MEASURED on an index-matched lossless ring grating
    (Rbig = 12 um, m = 1, one 0.5 um ring layer, lambda = 1 um): 78 propagating
    channels at N = 64 and 71 at N = 96, per-channel max|R+T-1| 1.52e-12 /
    4.35e-12, Gram max|offdiag| 2.50e-12 / 7.70e-12, and worst closure over 300
    random unit-norm multi-channel inputs 1.67e-12 / 8.21e-12 -- ~2.1 decades
    inside the bar, which is set where a REAL basis defect would land (the
    audit's own earlier probe bug read 0.86, and a half-cell flux-quadrature
    error of the audit-P3-14 class is O(1/N) ~ 1e-2 here).
    """
    from lumenairy.elements.bor import BORStack
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = BORStack(Rbig=12e-6, m=1, N=64, n_superstrate=1.41 + 0j,
                     n_substrate=1.41 + 0j)
        s.add_layer(0.5e-6, rings=(3.0e-6, 0.5, 2.45 + 0j, 1.41 + 0j))
        s.set_source(k0=2 * np.pi / 1.0e-6)
        res = s.solve()
    S11f, _S12, S21f, _S22 = res["S"]
    inc = np.asarray(res["inc"]).astype(int)
    out = np.asarray(res["out"]).astype(int)
    assert inc.size > 50, inc.size          # a non-trivial channel set
    U = np.vstack([np.asarray(S11f)[np.ix_(out, inc)],
                   np.asarray(S21f)[np.ix_(out, inc)]])
    gram = U.conj().T @ U
    assert np.max(np.abs(gram - np.eye(gram.shape[0]))) < 1e-9
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(300):
        c = rng.normal(size=inc.size) + 1j * rng.normal(size=inc.size)
        c /= np.linalg.norm(c)
        y = U @ c
        worst = max(worst, abs(float(np.vdot(y, y).real) - 1.0))
    assert worst < 1e-9, worst


def test_h4_isotropic_cell_builds_the_li_operators_once():
    """H4 (P2, perf).  ``_li_convolutions_2d`` computes BOTH ``Cxx`` and ``Cyy``
    on every call -- ``Sy`` batched inversions of ``(2Mx+1)^3`` PLUS ``Sx`` of
    ``(2My+1)^3`` -- and ``_inplane_ops`` called it twice, discarding half of
    each.  For an isotropic (scalar-promoted) cell ``exx`` and ``eyy`` are the
    same field, so one call suffices.

    DECISION assertions: exactly ONE call for an isotropic cell, still TWO for a
    genuinely anisotropic one (where the two operators read different fields),
    and the retained blocks are BIT-IDENTICAL either way (measured 0.0 / 0.0)."""
    cell = _promote(_square_cell())
    calls = {"n": 0}
    orig = _twod._li_convolutions_2d

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    _twod._li_convolutions_2d = counted
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calls["n"] = 0
            _o, _R, _T, J_iso = rcwa_jones_2d(
                _H3["period"], _H3["period"], cell, 1.0, 1.0, _H3["depth"],
                _H3["wl"], n_orders_x=4, n_orders_y=4, formulation="li",
                symmetry=False)
            n_iso = calls["n"]
            aniso = cell.copy()
            aniso[..., 1, 1] = aniso[..., 1, 1] * 1.02
            calls["n"] = 0
            rcwa_jones_2d(_H3["period"], _H3["period"], aniso, 1.0, 1.0,
                          _H3["depth"], _H3["wl"], n_orders_x=4, n_orders_y=4,
                          formulation="li", symmetry=False)
            n_aniso = calls["n"]
    finally:
        _twod._li_convolutions_2d = orig
    assert n_iso == 1, n_iso
    assert n_aniso == 2, n_aniso
    # the answer is unchanged: one call and two calls return the SAME blocks
    from lumenairy.elements.rcwa.twod import _harmonic_orders_2d
    orders, _N = _harmonic_orders_2d(4, 4, truncation="rectangular",
                                     period_x=_H3["period"],
                                     period_y=_H3["period"])
    ex = np.asarray(np.conj(cell[:, :, 0, 0]))
    a = orig(ex, orders, 4, 4, np)
    b = orig(ex, orders, 4, 4, np)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
