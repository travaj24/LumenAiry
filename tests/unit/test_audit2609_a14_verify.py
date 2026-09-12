"""VERIFY-A14 gates -- added by the independent re-verification of WP-A14
(``elements/rcwa``, ``elements/eme``, ``elements/bor``; audit findings H1-H6 and
G11 of ``AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11``).

These are properties the WP's own gate file does NOT pin, each measured on a
fixture WP-A14 did not use.  Every bar carries its oracle, the oracle's error
floor, the measured value on this build and the decades of gap on both sides
(``docs/TESTING_STANDARDS.md`` S1-S5).

Contents
--------
* the ``angle`` / ``theta`` alias mismatch is audible on the RCWA side too
  (the PMM mirror WP-A12 landed; the cross-suite resolution is UNCHANGED);
* the ``_WoodAnomaly`` control-flow re-entry forwards EVERY argument of every
  wrapped entry point, on both sides of the bracket;
* the Wood symmetric average beats the one-sided nudge on a mount with the
  anomaly in the SUBSTRATE (not the superstrate) and on a LOSSY metal mount,
  against an independent transfer-matrix (``expm``) oracle that shares no
  eigen-decomposition, mode-ordering or branch-cut machinery with the library;
* ``_sqrt_decay``'s new third conjunct is EXACTLY the propagating test
  ``Re(lam^2) < 0``, on a randomised population, and its NumPy and JAX bodies
  select the same set;
* the even-parity fold does NOT engage at oblique incidence for any
  formulation, and the isotropic single-build guard notices a ONE-PIXEL
  difference between ``exx`` and ``eyy``;
* ``guided_modes`` is AUDIBLE whenever it returns ``[]`` while the raw spectrum
  held a candidate -- through the ``reldiv`` and ``tail`` screens as well as
  through the guard band WP-A14 fixed (a defect found and fixed by this
  verification);
* ``radial_spectrum(return_modes=True)``'s eigenvectors are ``M``-orthonormal,
  pinned against the ANALYTIC Bessel normalisation (nothing in the suite pinned
  the scale, so the WP's deliberate change was unguarded in either direction).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest
from scipy.linalg import expm

from lumenairy.elements.rcwa import (
    RCWAStack,
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_jones_1d,
    rcwa_jones_2d,
)
from lumenairy.elements.rcwa._core import (
    _CUT_BAND_REL,
    _sqrt_decay,
    _WOOD_PAIR_STEP_REL,
)
from lumenairy.elements.rcwa.oned import _resolve_incidence
import lumenairy.elements.rcwa.twod as _twod


# ===========================================================================
# angle / theta alias: the RCWA mirror of the PMM WP-A12 notice
# ===========================================================================
def test_conflicting_angle_and_theta_is_audible_on_the_rcwa_side_too():
    """WP-A12 made ``pmm._core._resolve_incidence`` warn when ``angle=A`` and
    ``theta=T`` disagree.  The two resolvers are twins and
    ``test_v5_12_0_naming_aliases::test_set_source_theta_wins_consistent_across_suites``
    exists to pin that they agree, so the notice belongs on both.

    DECISION assertions (no numeric bar): the resolution is UNCHANGED
    (``theta`` wins, everywhere); a genuine disagreement warns and names both
    values; the three non-ambiguous spellings stay silent.  The silent arms are
    what make this two-sided: a warning on ``theta=``-only would fire on the
    ordinary call, since ``angle`` then sits at its ``0.0`` / ``None`` default
    and is indistinguishable from an explicit zero.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = _resolve_incidence(0.9, 0.25)
    assert got == 0.25                                    # theta still wins
    assert any("DISAGREE" in str(w.message) for w in rec), \
        [str(w.message) for w in rec]
    assert any("rcwa:" in str(w.message) for w in rec)
    for args in ((0.0, 0.25), (0.25, 0.25), (0.9, None), (None, 0.25),
                 (None, None)):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _resolve_incidence(*args)
        assert not rec, (args, [str(w.message) for w in rec])
    # end to end through both public surfaces, resolution unchanged
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        o_both, R_both, _T = rcwa_efficiency_1d(
            1e-6, 2.04, 1.0, 1.5, 1.0, 0.3e-6, 0.5, 0.633e-6,
            angle=0.7, theta=0.25, n_orders=5)
    assert any("DISAGREE" in str(w.message) for w in rec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R_theta, _T = rcwa_efficiency_1d(
            1e-6, 2.04, 1.0, 1.5, 1.0, 0.3e-6, 0.5, 0.633e-6,
            theta=0.25, n_orders=5)
        _o, R_angle, _T = rcwa_efficiency_1d(
            1e-6, 2.04, 1.0, 1.5, 1.0, 0.3e-6, 0.5, 0.633e-6,
            angle=0.7, n_orders=5)
    assert np.array_equal(R_both, R_theta)       # theta won, bit for bit
    assert not np.array_equal(R_both, R_angle)   # ... and angle really differs
    grid = np.where(np.arange(64) < 32, 4.0, 1.0).astype(complex)
    st = RCWAStack(1e-6, n_substrate=1.5, n_superstrate=1.0, n_orders=5)
    st.add_layer(0.3e-6, eps_cell=grid)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        st.set_source(0.633e-6, angle=0.7, theta=0.25)
    assert any("DISAGREE" in str(w.message) for w in rec)
    assert st._source["theta"] == 0.25


# ===========================================================================
# H2 -- the _WoodAnomaly re-entry must forward EVERY argument
# ===========================================================================
_WL = 1.0e-6
_P = 1.0e-6
_LO = _WL * (1.0 - _WOOD_PAIR_STEP_REL)
_HI = _WL * (1.0 + _WOOD_PAIR_STEP_REL)

_CELL2 = np.full((48, 48), 1.0 + 0j)
_CELL2[12:36, 12:36] = 6.25 + 0j
_TEN = np.zeros((48, 48, 3, 3), dtype=complex)
for _i in range(3):
    _TEN[..., _i, _i] = _CELL2
_TEN[..., 0, 1] = 0.15 * (_CELL2 - 1.0)
_TEN[..., 1, 0] = 0.15 * (_CELL2 - 1.0)


def _flat(x):
    out = []
    if isinstance(x, tuple):
        for e in x:
            out += _flat(e)
    elif hasattr(x, "__array__"):
        out.append(np.asarray(x).ravel())
    elif isinstance(x, (int, float, complex)):
        out.append(np.array([x]))
    return out


@pytest.mark.parametrize("label,fn,args,kw", [
    ("efficiency_1d, positional + tm/laurent/n_orders=9",
     rcwa_efficiency_1d, (_P, 2.04, 1.0, 1.0, 1.0, 1.0e-6, 0.37, _WL),
     dict(polarization="tm", n_orders=9, formulation="laurent")),
    ("efficiency_1d, all-keyword + te/li/n_orders=13",
     rcwa_efficiency_1d, (),
     dict(period=_P, n_ridge=2.04, n_groove=1.0, n_substrate=1.0,
          n_superstrate=1.0, depth=1.0e-6, duty_cycle=0.37, wavelength=_WL,
          polarization="te", n_orders=13, formulation="li")),
    ("efficiency_1d, ASR eta=0.5 / samples=4096",
     rcwa_efficiency_1d, (_P, 2.04, 1.0, 1.0, 1.0, 1.0e-6, 0.5, _WL),
     dict(polarization="tm", n_orders=11, formulation="li", asr_eta=0.5,
          asr_samples=4096)),
    ("jones_1d, tensor ridge + return_jones_transmission",
     rcwa_jones_1d,
     (_P, np.diag([2.4 ** 2, 2.1 ** 2, 2.1 ** 2]).astype(complex),
      np.eye(3, dtype=complex), 1.0, 1.0, 0.4e-6, 0.5, _WL),
     dict(n_orders=9, formulation="li", return_jones_transmission=True)),
    ("efficiency_2d, li, 3x2, tm, symmetry off",
     rcwa_efficiency_2d, (_P, _P, _CELL2, 1.0, 1.0, 0.3e-6, _WL),
     dict(n_orders_x=3, n_orders_y=2, polarization="tm", formulation="li",
          symmetry=False)),
    ("efficiency_2d, laurent, symmetry ON (even fold)",
     rcwa_efficiency_2d, (_P, _P, _CELL2, 1.0, 1.0, 0.3e-6, _WL),
     dict(n_orders_x=3, n_orders_y=3, polarization="te",
          formulation="laurent", symmetry=True)),
    ("jones_2d, fff_nv + allow_nonseparable_nv",
     rcwa_jones_2d, (_P, _P, _TEN, 1.0, 1.0, 0.3e-6, _WL),
     dict(n_orders_x=3, n_orders_y=3, formulation="fff_nv",
          allow_nonseparable_nv=True, symmetry=False)),
    ("jones_2d, li + symmetry ON",
     rcwa_jones_2d, (_P, _P, _TEN, 1.0, 1.0, 0.3e-6, _WL),
     dict(n_orders_x=3, n_orders_y=3, formulation="li", symmetry=True)),
])
def test_wood_symmetric_reentry_forwards_every_argument(label, fn, args, kw):
    """H2.  ``_wood_symmetric`` re-enters the decorated entry point through
    ``inspect.signature(fn).bind(*args, **kwargs)`` and a mutated
    ``BoundArguments``.  That is a deliberate alternative to each entry point
    re-listing its ~20 arguments, and the WHOLE point of it is that nothing can
    be dropped -- so it needs a gate that would notice if something were.

    ORACLE: the two legs run EXPLICITLY, with the same argument list plus
    ``_wl_eff=lo`` / ``_wl_eff=hi`` (which bypasses the detection and the
    re-entry entirely), averaged by hand.  A dropped or reordered argument on
    either leg changes that leg's answer and breaks the identity.

    BAR: EXACT equality (0.0).  Both sides run the same double-precision code on
    the same inputs, so anything but bit-identity is a real difference; MEASURED
    0.0 on every arm.  The arms deliberately vary the arguments most likely to
    be lost: a purely positional call, a purely keyword call, ASR (two extra
    kwargs), the tensor entries, ``symmetry`` (which selects a DIFFERENT solve
    path) and ``allow_nonseparable_nv``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = fn(*args, **kw)
        lo = fn(*args, **{**kw, "_wl_eff": _LO})
        hi = fn(*args, **{**kw, "_wl_eff": _HI})
    f_got, f_lo, f_hi = _flat(got), _flat(lo), _flat(hi)
    assert len(f_got) == len(f_lo) == len(f_hi)
    # the bracket must actually have fired, or the test is vacuous
    assert any(not np.array_equal(a, b) for a, b in zip(f_lo, f_hi)), label
    for a, u, v in zip(f_got, f_lo, f_hi):
        if u.dtype.kind in "iu":
            assert np.array_equal(a, u), label
            continue
        man = 0.5 * (u.astype(complex) + v.astype(complex))
        assert np.max(np.abs(a.astype(complex) - man)) == 0.0, label


def test_wood_symmetric_reentry_forwards_stack_arguments():
    """H2, the ``RCWAStack`` arm: ``_solve_once`` is a METHOD, so the re-entry
    has to carry ``self`` as well.  Same oracle and same exact bar."""
    cell = np.full((48, 48), 1.0 + 0j)
    cell[12:36, 12:36] = 6.25 + 0j
    st = RCWAStack(_P, period_y=_P, n_orders=3, n_orders_y=2,
                   n_superstrate=1.0, n_substrate=1.0)
    st.add_layer(0.3e-6, eps_cell=cell)
    st.set_source(_WL, theta=0.0, phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.solve()
        lo = st._solve_once(_wl_eff=_LO)
        hi = st._solve_once(_wl_eff=_HI)
    _o, R, T = res.efficiencies()
    _o, Rl, Tl = lo.efficiencies()
    _o, Rh, Th = hi.efficiencies()
    assert np.max(np.abs(Rl - Rh)) > 0.0          # the bracket really fired
    assert np.max(np.abs(R - 0.5 * (Rl + Rh))) == 0.0
    assert np.max(np.abs(T - 0.5 * (Tl + Th))) == 0.0
    assert isinstance(res.wl_eff, tuple) and res.wl_eff == (_LO, _HI)


# ===========================================================================
# H2 -- an independent transfer-matrix oracle on TWO mounts WP-A14 did not use
# ===========================================================================
def _binary_coeffs(c_lo, c_hi, duty, nk):
    k = np.arange(-(nk - 1), nk).astype(float)
    ramp = duty * np.sinc(k * duty) * np.exp(-1j * np.pi * k * duty)
    return c_lo * (k == 0).astype(complex) + (c_hi - c_lo) * ramp


def _toep(c, M):
    N = 2 * M + 1
    ctr = (c.shape[0] - 1) // 2
    i = np.arange(N)
    return c[ctr + (i[:, None] - i[None, :])]


def _sq(x):
    r = np.sqrt(np.asarray(x, dtype=complex))
    return np.where(r.imag < 0, -r, r)


def _oracle_expm(period, n_ridge, n_groove, n_sub, n_sup, depth, duty, wl,
                 pol="te", M=5):
    """1-D efficiencies from the EXPONENTIAL of the first-order Maxwell system.

    Deliberately shares nothing with the library beyond Maxwell: there is no
    layer eigendecomposition (so no mode ordering, no ``_sqrt_decay`` branch
    choice, no ``_inv_lam`` floor), no S-matrix and no Redheffer star -- the
    layer is a single ``scipy.linalg.expm`` of ``[[0, I], [Kx^2 - E, 0]]``
    (TE) / ``i [[0, I - Kx E^-1 Kx], [(1/eps)_toep^-1, 0]]`` (TM), matched
    directly to the half-space plane waves.  ``M`` stays small because the
    matrix exponential of an evanescent order grows like ``exp(M k0 d)``.
    """
    k0 = 2 * np.pi / wl
    N = 2 * M + 1
    m = np.arange(-M, M + 1)
    eps_r, eps_g = complex(n_ridge) ** 2, complex(n_groove) ** 2
    eps_I, eps_II = complex(n_sup) ** 2, complex(n_sub) ** 2
    alpha = m * (wl / period)
    Kx = np.diag(alpha.astype(complex))
    gI, gII = _sq(eps_I - alpha ** 2), _sq(eps_II - alpha ** 2)
    g0 = _sq(eps_I).real
    E = _toep(_binary_coeffs(eps_g, eps_r, duty, N), M)
    Ei = _toep(_binary_coeffs(1 / eps_g, 1 / eps_r, duty, N), M)
    I = np.eye(N, dtype=complex)
    Zc = np.zeros((N, N), dtype=complex)
    delta = (m == 0).astype(complex)
    dp = k0 * depth
    if pol == "te":
        T = expm(np.block([[Zc, I], [Kx @ Kx - E, Zc]]) * dp)
        T11, T12, T21, T22 = T[:N, :N], T[:N, N:], T[N:, :N], T[N:, N:]
        GI, GII = np.diag(gI), np.diag(gII)
        P = T21 - 1j * GII @ T11
        Q = T22 - 1j * GII @ T12
        r = -np.linalg.solve(P - 1j * Q @ GI, (P + 1j * Q @ GI) @ delta)
        t = T11 @ (delta + r) + 1j * T12 @ (GI @ (delta - r))
        DEr = np.real(gI / g0) * np.abs(r) ** 2
        DEt = np.real(gII / g0) * np.abs(t) ** 2
    else:
        A = 1j * np.block([[Zc, I - Kx @ np.linalg.inv(E) @ Kx],
                           [np.linalg.inv(Ei), Zc]])
        T = expm(A * dp)
        T11, T12, T21, T22 = T[:N, :N], T[:N, N:], T[N:, :N], T[N:, N:]
        G, H = np.diag(gI / eps_I), np.diag(gII / eps_II)
        U = (H @ T21 - T11) @ G
        V = H @ T22 - T12
        r = -np.linalg.solve(V - U, (U + V) @ delta)
        t = T21 @ (G @ (delta - r)) + T22 @ (delta + r)
        DEr = np.real(gI / eps_I) / (g0 / eps_I.real) * np.abs(r) ** 2
        DEt = np.real(gII / eps_II) / (g0 / eps_I.real) * np.abs(t) ** 2
    DEr = np.where(np.real(gI) > 1e-14, DEr, 0.0)
    DEt = np.where(np.real(gII) > 1e-14, DEt, 0.0)
    return m, np.real(DEr), np.real(DEt)


# (mount, pol, min improvement factor of the symmetric average over the
#  one-sided nudge, absolute bar on the symmetric average's R0 error)
# The improvement is NOT the Moharam mount's 60x everywhere: measured
# 5.9x / 9.8x (substrate anomaly, TE / TM) and 24x / 1.05x (lossy Ag), because
# the sqrt(delta) coefficient and the sub-leading term differ per mount.  The
# bars below are set from the MEASURED family with a decade of slack, not from
# the WP's single fixture.
_ALT_MOUNTS = [
    ("substrate anomaly n_sub=1.5", dict(
        period=1.0e-6, n_ridge=2.04, n_groove=1.0, n_sub=1.5, n_sup=1.0,
        depth=0.4e-6, duty=0.5, wl=1.5e-6), "te", 2.0, 3e-4),
    ("substrate anomaly n_sub=1.5", dict(
        period=1.0e-6, n_ridge=2.04, n_groove=1.0, n_sub=1.5, n_sup=1.0,
        depth=0.4e-6, duty=0.5, wl=1.5e-6), "tm", 2.0, 3e-4),
    ("lossy Ag ridge", dict(
        period=1.0e-6, n_ridge=0.135 + 3.99j, n_groove=1.0, n_sub=1.0,
        n_sup=1.0, depth=0.2e-6, duty=0.5, wl=1.0e-6), "te", 2.0, 1e-4),
]


@pytest.mark.parametrize("name,mt,pol,min_gain,abs_bar", _ALT_MOUNTS)
def test_h2_symmetric_average_beats_the_one_sided_nudge_off_the_wp_fixture(
        name, mt, pol, min_gain, abs_bar):
    """H2 (P1) on mounts WP-A14 did NOT use: the anomaly in the SUBSTRATE
    (so the grazing pair is in TRANSMISSION, not reflection) and a LOSSY metal
    ridge (so the tight lossless closure clause is disarmed).

    ORACLE: :func:`_oracle_expm` at the EXACT requested wavelength -- an
    independent transfer-matrix solve with no eigendecomposition, no S-matrix
    and no branch-cut machinery in common with the library.  Its own error
    floor at ``M = 5`` is its agreement with the audit's independent 4N
    boundary-match oracle on the same mounts: MEASURED 6.4e-15 / 4.3e-15 /
    5.3e-15 max|dR| and its own energy closure |dE| <= 1.3e-14 on the lossless
    mounts -- 10 decades below the bars here.

    BARS (two-sided).  MEASURED at ``M = 5`` on this build, |R0 - oracle|:
    substrate TE 1.9e-05 symmetric vs 1.1e-04 one-sided; substrate TM 1.8e-05
    vs 1.7e-04; lossy Ag TE 3.4e-06 vs 8.1e-05.  The gain bar (2x) sits ~0.5
    decade below the smallest measured gain (5.7x) and far above 1.0, which is
    what "no improvement" would read; the absolute bar sits ~1 decade above the
    measured symmetric error and ~0.5 decade BELOW the measured one-sided
    error, so a silent revert to the one-sided rule fails it.
    """
    M = 5
    _m, Ro, To = _oracle_expm(mt["period"], mt["n_ridge"], mt["n_groove"],
                              mt["n_sub"], mt["n_sup"], mt["depth"],
                              mt["duty"], mt["wl"], pol=pol, M=M)

    def lib(wl):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T = rcwa_efficiency_1d(
                mt["period"], mt["n_ridge"], mt["n_groove"], mt["n_sub"],
                mt["n_sup"], mt["depth"], mt["duty"], wl, polarization=pol,
                n_orders=M, formulation="li")
        return np.asarray(R), np.asarray(T)

    R_sym, T_sym = lib(mt["wl"])
    R_one, _T_one = lib(mt["wl"] * (1.0 + 1e-7))        # the pre-fix rule
    err_sym = abs(R_sym[M] - Ro[M])
    err_one = abs(R_one[M] - Ro[M])
    assert err_one > 10.0 * abs_bar / 30.0, (
        f"{name} {pol}: the one-sided nudge is no longer visibly wrong "
        f"({err_one:.3e}) -- fixture stale")
    assert err_sym < abs_bar, f"{name} {pol}: {err_sym:.3e}"
    assert err_one / err_sym > min_gain, (
        f"{name} {pol}: gain {err_one / err_sym:.2f}x "
        f"(sym {err_sym:.3e}, one-sided {err_one:.3e})")
    # the closure is untouched by the average (it is exact for a lossless
    # mount and equals the absorptance for the lossy one -- both to 1e-11)
    assert abs((R_sym.sum() + T_sym.sum())
               - (Ro.sum() + To.sum())) < 1e-4


# ===========================================================================
# G11 -- the third conjunct IS the propagating test, and JAX agrees
# ===========================================================================
def test_g11_flip_set_is_exactly_the_propagating_subset_of_the_band():
    """G11 (P3).  The new conjunct ``Im(r)^2 > Re(r)^2`` is algebraically
    identical to ``Re(r^2) < 0`` -- i.e. to "this mode is PROPAGATING" -- so the
    docstring's promise ("a flipped mode is by construction a propagating one")
    becomes a theorem.  This gate states that as a SET identity on a randomised
    population rather than on the two hand-built counterexamples the WP pins.

    ORACLE: the definition itself.  For a root ``r`` returned by the principal
    ``sqrt``, the mode is propagating iff ``Re(lam^2) = Re(r)^2 - Im(r)^2 < 0``.
    The population spans 11 decades of magnitude and both sides of the
    45-degree line, and the fixture asserts that BOTH outcomes occur (flips and
    non-flips) so the identity cannot go vacuous.

    BAR: exact set equality, and every flipped entry must satisfy
    ``Re(lam^2) < 0`` with NO tolerance (both quantities come from the same
    doubles).  MEASURED on this build: 5000 draws, the flip set matches the
    predicate exactly, and the band-only (round-1) predicate would have flipped
    strictly more.
    """
    rng = np.random.default_rng(20260912)
    n = 5000
    mag = 10.0 ** rng.uniform(-11.0, 1.0, n)
    ang = rng.uniform(-np.pi, np.pi, n)
    lam2 = mag * np.exp(1j * ang)
    # a slice of the population deliberately ON the cut (lam^2 negative real
    # with a backward-error imaginary part), where the band is meant to fire
    k = n // 4
    lam2[:k] = -(10.0 ** rng.uniform(-6.0, 1.0, k)) \
        - 1j * (10.0 ** rng.uniform(-30.0, -14.0, k))
    # ... and a slice of NEAR-ZERO EVANESCENT roots (the audit's own G11
    # counterexample family, lam^2 a tiny POSITIVE real with a backward-error
    # negative imaginary part).  These sit inside ``band * scale`` because the
    # band is relative to the spectrum's TOP, which is what made the round-1
    # predicate flip them; the new conjunct must not.
    lam2[k:2 * k] = (10.0 ** rng.uniform(-22.0, -17.0, k)) \
        - 1j * (10.0 ** rng.uniform(-34.0, -28.0, k))
    out = _sqrt_decay(lam2)
    prin = np.sqrt(lam2.astype(complex))
    flipped = ~np.isclose(out, prin, rtol=0.0, atol=0.0)
    scale = max(float(np.max(np.abs(prin))), 1.0)
    band = np.abs(prin.real) <= _CUT_BAND_REL * scale
    predicate = band & (prin.imag < 0) & (prin.imag ** 2 > prin.real ** 2)
    assert np.array_equal(flipped, predicate)
    assert flipped.any() and (~flipped).any()          # not vacuous
    # every flipped root is PROPAGATING, with no tolerance
    assert np.all(prin.real[flipped] ** 2 - prin.imag[flipped] ** 2 < 0.0)
    # and no flipped root is handed back with Re(lam) < 0
    assert np.all(out.real >= 0.0) or np.all(out[flipped].imag >= 0.0)
    # the round-1 predicate (band only) was strictly larger on this population
    round1 = band & (prin.imag < 0)
    assert round1.sum() > predicate.sum()
    assert np.all(prin.real[round1 & ~predicate] ** 2
                  >= prin.imag[round1 & ~predicate] ** 2)


def test_g11_numpy_and_jax_bodies_select_the_same_flips():
    """G11 (P3).  ``_sqrt_decay`` is backend-generic and the JAX twin is the
    SAME body under ``jax.numpy``; if the predicate ever diverged between them
    a stack would decay on one backend and grow on the other.

    ORACLE: the NumPy flip SET.  BAR: exact set equality, and a value bar of
    1e-12 -- MEASURED 8.0e-14, which is XLA's ``sqrt`` differing from NumPy's
    in the last bits (the same 8.0e-14 separates the raw ``sqrt``s, so it is
    not this predicate's doing) and is 1.5 decades inside the bar.
    """
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    rng = np.random.default_rng(7)
    n = 1200
    lam2 = (10.0 ** rng.uniform(-11.0, 1.0, n)) * np.exp(
        1j * rng.uniform(-np.pi, np.pi, n))
    k = n // 3
    lam2[:k] = -(10.0 ** rng.uniform(-6.0, 1.0, k)) \
        - 1j * (10.0 ** rng.uniform(-30.0, -14.0, k))
    ref = _sqrt_decay(lam2)
    got = np.asarray(_sqrt_decay(jnp.asarray(lam2), xp=jnp))
    prin = np.sqrt(lam2.astype(complex))
    f_np = ~np.isclose(ref, prin, rtol=0.0, atol=0.0)
    f_jx = np.sign(got.imag) != np.sign(prin.imag)
    assert f_np.any()
    assert np.array_equal(f_np, f_jx)
    assert np.max(np.abs(ref - got)) < 1e-12


# ===========================================================================
# H4 -- fold engagement conditions and the isotropic single-build guard
# ===========================================================================
_H4_CELL = np.full((96, 96), 2.25 + 0j)
_H4_CELL[24:72, 24:72] = 6.25 + 0j


def _promote(c):
    t = np.zeros(c.shape + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = c
    return t


@pytest.mark.parametrize("formulation", ["laurent", "li", "fff_nv"])
def test_h4_even_fold_does_not_engage_at_oblique_incidence(formulation):
    """H4 (P2).  The fold is only valid at NORMAL incidence -- it folds the
    order set about the symmetry centre, which a non-zero ``kx0``/``ky0``
    destroys.  Generalising the fold from ``'laurent'`` to every in-plane
    formulation must NOT have loosened that precondition.

    DECISION assertion: ``_symmetric_cascade_rt`` is not called at all at
    ``theta = 20 deg``, for any formulation, with ``symmetry='auto'`` AND with
    ``symmetry=True`` (an explicit request must still fall through, not
    silently produce the wrong basis).  The normal-incidence arm is asserted in
    the same test so the counter is known to work.
    """
    cell = _promote(_H4_CELL)
    calls = []
    orig = _twod._symmetric_cascade_rt
    _twod._symmetric_cascade_rt = lambda *a, **k: (calls.append(1)
                                                   or orig(*a, **k))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for sym in (True, "auto"):
                calls.clear()
                rcwa_jones_2d(0.5e-6, 0.5e-6, cell, 1.0, 1.0, 0.3e-6,
                              0.633e-6, n_orders_x=3, n_orders_y=3,
                              theta=np.deg2rad(20.0), phi=0.0,
                              formulation=formulation, symmetry=sym)
                assert calls == [], (formulation, sym, len(calls))
            calls.clear()
            rcwa_jones_2d(0.5e-6, 0.5e-6, cell, 1.0, 1.0, 0.3e-6, 0.633e-6,
                          n_orders_x=3, n_orders_y=3, formulation=formulation,
                          symmetry=True)
            assert len(calls) == 1, (formulation, calls)
    finally:
        _twod._symmetric_cascade_rt = orig


def test_h4_isotropic_single_build_guard_sees_a_one_pixel_difference():
    """H4 (P2, perf).  ``_inplane_ops`` now builds the Li operators ONCE when
    ``exx`` and ``eyy`` are the same field.  The guard is ``exx is eyy`` then
    ``all(exx == eyy)``; if it ever weakened to a tolerance or a shape/dtype
    test, an anisotropic cell would silently get ``Cyy`` built from ``exx``.

    DECISION assertion: a cell whose ``eyy`` differs from ``exx`` in ONE PIXEL
    by one part in 1e-9 must still take the TWO-call path, and must give a
    numerically different answer from the isotropic cell (so the distinction is
    not cosmetic).  MEASURED: 2 calls, max|dJ| 2.0e-12 between the two cells --
    small, which is exactly why a tolerance-based guard would be wrong.
    """
    cell = _promote(_H4_CELL)
    tweaked = cell.copy()
    tweaked[0, 0, 1, 1] = tweaked[0, 0, 1, 1] * (1.0 + 1e-9)
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
                0.5e-6, 0.5e-6, cell, 1.0, 1.0, 0.3e-6, 0.633e-6,
                n_orders_x=4, n_orders_y=4, formulation="li", symmetry=False)
            n_iso = calls["n"]
            calls["n"] = 0
            _o, _R, _T, J_tw = rcwa_jones_2d(
                0.5e-6, 0.5e-6, tweaked, 1.0, 1.0, 0.3e-6, 0.633e-6,
                n_orders_x=4, n_orders_y=4, formulation="li", symmetry=False)
            n_tw = calls["n"]
    finally:
        _twod._li_convolutions_2d = orig
    assert n_iso == 1, n_iso
    assert n_tw == 2, n_tw
    assert np.max(np.abs(J_iso - J_tw)) > 0.0


# ===========================================================================
# H1 -- the empty list must be AUDIBLE through every filter, not just the band
# ===========================================================================
_LAM_F = 1.55e-6
_K0_F = 2 * np.pi / _LAM_F


@pytest.mark.parametrize("n1,n2,V,N,label", [
    (3.48, 1.44, 2.0, 200, "Si/SiO2 dn=2.04, V=2.0 (TAIL screen)"),
    (2.00, 1.44, 2.0, 300, "SiN/SiO2 dn=0.56, V=2.0 (TAIL screen)"),
    (1.45, 1.44, 2.6, 200, "dn=0.010, V=2.6, m=2 (TAIL screen)"),
])
def test_h1_an_empty_guided_mode_list_is_never_silent(n1, n2, V, N, label):
    """H1 (P1), the half WP-A14 left open.  Its fix made the guard band
    contrast-invariant and added a notice for the case the BAND empties -- but
    the band is one of THREE doors, and on a high-contrast or slightly
    over-moded fibre it is the ``tail_tol`` radiation screen (``Rbig`` too
    small for the mode's cladding tail) that empties the list.  Those cases
    still returned a SILENT ``[]``, which is the exact user-visible failure
    H1 is about.

    ORACLE: ``fiber_oracle.fiber_modes`` -- the exact hybrid HE/EH 4x4
    boundary-match determinant, which shares no code with the FD vector
    eigensolver.  It says a guided mode EXISTS on each fixture below.

    DECISION assertion (no numeric bar): when ``guided_modes`` returns ``[]``
    on a structure the exact oracle says guides, it must emit exactly one
    warning, and that warning must name the rejecting filter so the caller
    knows which knob to turn.  MEASURED before the fix: 0 warnings on all
    three fixtures (and on Si/SiO2 at V = 2.0 the raw spectrum held
    n_eff = 1.882 against the oracle's 1.8469, rejected with tail = 1.00).
    The non-vacuity arm asserts the oracle really does find a mode.
    """
    from lumenairy.elements.bor.coupled_radial_eigensolver import guided_modes
    from lumenairy.elements.bor.fiber_oracle import fiber_modes
    m = 2 if V > 2.405 and abs(n1 - n2) < 0.05 else 1
    a = V / (_K0_F * np.sqrt(n1 ** 2 - n2 ** 2))
    exact = fiber_modes(m, a, n1 ** 2, n2 ** 2, _K0_F)
    assert exact.size >= 1, f"{label}: the oracle finds no mode -- fixture stale"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gm = guided_modes(m, a, 6.0 * a, N, n1 ** 2, n2 ** 2, _K0_F)
    if gm:                       # a build that DOES find it needs no warning
        return
    msgs = [str(w.message) for w in rec
            if "guided_modes: no mode survived" in str(w.message)]
    assert len(msgs) == 1, (label, [str(w.message) for w in rec])
    assert "rejected because" in msgs[0], msgs[0]
    assert ("SPURIOUS-mode screen" in msgs[0]
            or "RADIATION screen" in msgs[0]
            or "guard band" in msgs[0]), msgs[0]


def test_h1_a_populated_guided_mode_list_stays_quiet():
    """The other side of the same claim: the notice must NOT fire when modes
    ARE returned, or it would be noise on every ordinary call.  Fixtures: the
    four legacy suite geometries (eps 6 / 2) and the weakly-guiding V = 2.4
    fibre WP-A14 added."""
    from lumenairy.elements.bor.coupled_radial_eigensolver import guided_modes
    for args in ((1, 1.0, 8.0, 600, 6.0, 2.0, 2.0),
                 (1, 1.0, 6.0, 300, 6.0, 2.0, 2.0)):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            gm = guided_modes(*args[:6], args[6])
        assert gm, args
        assert not [w for w in rec
                    if "no mode survived" in str(w.message)], args
    n1, n2 = 1.45, 1.44
    a = 2.4 / (_K0_F * np.sqrt(n1 ** 2 - n2 ** 2))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gm = guided_modes(1, a, 6.0 * a, 150, n1 ** 2, n2 ** 2, _K0_F)
    assert len(gm) >= 1
    assert not [w for w in rec if "no mode survived" in str(w.message)]


# ===========================================================================
# H4(c) -- the eigenvector NORMALISATION the WP changed, pinned analytically
# ===========================================================================
def test_h4_radial_spectrum_modes_are_M_orthonormal_against_bessel():
    """H4 (P2).  ``radial_spectrum(return_modes=True)`` moved from
    ``np.linalg.eig`` (unit 2-norm eigenvectors) to ``scipy.linalg.eigh(A, M)``
    (``M``-orthonormal, i.e. ``INT_0^R r psi^2 dr = 1``).  That is a PUBLIC
    scale change, and BOTH existing consumers
    (``test_radial_eigensolver::test_eigenfunctions_match_bessel_profiles``,
    which least-squares-fits a scale, and
    ``test_niche_audit_w6_bor::test_w6_oracle_radial_spectrum_rdr_orthonormal_and_regular``,
    which normalises the Gram) are scale-INVARIANT -- so nothing pinned it in
    either direction.

    ORACLE: the ANALYTIC eigenfunctions of the unit disk.  For Dirichlet bc the
    n-th m-th-order radial mode is ``c_n J_m(j_{m,n} r)`` and
    ``INT_0^1 r J_m(j r)^2 dr = J_{m+1}(j)^2 / 2``, so an ``M``-orthonormal
    eigenvector has ``c_n = sqrt(2) / |J_{m+1}(j_{m,n})|`` exactly.  The oracle
    floor is the SEM's own discretisation error, which the eigenvalues show to
    be ~1e-13 relative at degree 8 / 12 elements.

    BAR 1e-9 relative on the profile.  MEASURED: 1.0e-13 / 2.2e-13 / 3.8e-12 /
    4.1e-11 for modes 0-3 (the higher modes are less resolved) -- 1.4 decades
    inside the bar for the worst mode.  The unit-2-norm scale it replaced is
    14.0-17.7x larger here, i.e. 10 decades outside the bar, so a silent revert
    fails this immediately.
    """
    from scipy.special import jn_zeros, jv
    from lumenairy.elements.bor.radial_eigensolver import radial_spectrum
    m, n_low = 1, 4
    _w, vec, rg = radial_spectrum(m, 1.0, 8, 12, bc="dirichlet", n_low=n_low,
                                  return_modes=True)
    jz = jn_zeros(m, n_low)
    norms = np.linalg.norm(vec[:, :n_low], axis=0)
    assert np.all(norms > 5.0)          # NOT unit 2-norm (measured 14.0-17.7)
    for n in range(n_low):
        exact = (np.sqrt(2.0) / abs(jv(m + 1, jz[n]))) * jv(m, jz[n] * rg)
        col = vec[:, n]
        sgn = 1.0 if float(np.real(np.vdot(exact, col))) >= 0 else -1.0
        rel = (np.max(np.abs(sgn * col - exact))
               / np.max(np.abs(exact)))
        assert rel < 1e-9, (n, rel)
    assert np.all(vec[0, :] == 0.0)     # axis regularity, m != 0, exact


# ===========================================================================
# H6 -- the passive guard must not RAISE on an input the solver accepted
# ===========================================================================
def test_h6_passive_guard_does_not_break_array_valued_indices():
    """H6 (P3).  The 1-D passive guard packed its two layer permittivities into
    one ``np.array([eps_ridge, eps_groove])``.  That pack is evaluated OUTSIDE
    ``_passive_media``'s own ``except (TypeError, ValueError)``, so a shape
    mismatch between the two indices -- ``n_ridge = np.array([2.04+0j])`` with a
    scalar ``n_groove``, which the solver otherwise accepts -- raised
    ``ValueError: setting an array element with a sequence`` from the ENERGY
    GUARD, after a complete and correct solve.

    DECISION assertions: every scalar/array spelling of the two indices returns,
    and they all return the SAME number (so the guard is not changing the
    answer); and the predicate is unchanged by unpacking -- ``_passive_media``
    takes the max over its ``eps_arrays`` one at a time, so the packed and the
    separate forms are the same boolean on all four clause combinations.
    """
    from lumenairy.elements.rcwa._core import _passive_media
    spellings = [(2.04, 1.0),
                 (np.array([2.04 + 0j]), 1.0),
                 (np.array([2.04 + 0j]), np.array([1.0 + 0j])),
                 (np.array(2.04 + 0j), np.array(1.0 + 0j))]
    vals = []
    for nr, ng in spellings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, _T = rcwa_efficiency_1d(
                0.5e-6, nr, ng, 1.5, 1.0, 0.3e-6, 0.5, 0.633e-6,
                n_orders=5, polarization="tm")
        vals.append(float(np.ravel(R)[5]))
    assert vals[0] > 1e-6                              # a non-trivial answer
    for v in vals[1:]:
        assert v == vals[0]                            # bit-identical
    # the predicate itself is unchanged by unpacking
    for a, b in ((2.25 + 0j, 1.0 + 0j), (2.25 - 0.3j, 1.0 + 0j),
                 (2.25 + 0.3j, 1.0 + 0j), (1.0 + 0j, 2.25 + 0.1j)):
        packed = _passive_media(1.0 + 0j, 2.25 + 0j, np.array([a, b]))
        apart = _passive_media(1.0 + 0j, 2.25 + 0j, np.asarray(a),
                               np.asarray(b))
        assert packed == apart, (a, b)


# ===========================================================================
# H1 -- a PARTIAL guided_modes result must be loud too (VERIFY-A14 V1-residual)
# ===========================================================================
def test_h1_a_short_guided_mode_list_is_never_silent():
    """H1 (P1), the last quiet door.  An EMPTY list is now audible through
    every filter, but a list that is merely SHORT was not: Si/SiO2
    (``dn = 2.04``) at V = 4.0 returned ONE mode while the exact hybrid
    characteristic equation has THREE, with no signal at all.

    ORACLE: :func:`_step_index_root_census` -- a sign-change scan of
    ``fiber_oracle.fiber_det``, the 4x4 Bessel boundary-match determinant,
    which shares no code with the finite-difference vector eigensolver it
    counts against.  Its own floor is the scan resolution
    ``(n_core - n_clad) / 2000`` in ``n_eff``, and its error is ONE-SIDED
    (it can only under-count), so the notice cannot fire spuriously -- which is
    what makes a DECISION assertion legitimate here rather than a numeric bar.

    CROSS-CHECK of the oracle itself (so the gate is not pinning a
    miscounting scan): the census is compared against the BISECTING
    :func:`~lumenairy.elements.bor.fiber_oracle.fiber_modes` on twelve
    fixtures spanning m = 0..5, V = 1.8..9 and three index systems -- MEASURED
    identical counts on all twelve.

    MEASURED before: Si/SiO2 V = 4.0 returned 1 mode, 0 warnings.
    """
    from lumenairy.elements.bor.coupled_radial_eigensolver import (
        guided_modes, _step_index_root_census)
    from lumenairy.elements.bor.fiber_oracle import fiber_modes
    # (a) the census agrees with the bisecting oracle everywhere
    for m, n1, n2, V in [(1, 3.48, 1.44, 2.0), (1, 3.48, 1.44, 4.0),
                         (1, 1.45, 1.44, 2.4), (0, 1.45, 1.44, 2.6),
                         (2, 1.45, 1.44, 2.6), (1, 2.00, 1.44, 2.0),
                         (0, 3.48, 1.44, 4.0), (2, 3.48, 1.44, 4.0),
                         (3, 3.48, 1.44, 6.0), (1, 1.45, 1.445, 2.4),
                         (0, 1.45, 1.44, 1.8), (5, 3.48, 1.44, 9.0)]:
        a = V / (_K0_F * np.sqrt(n1 ** 2 - n2 ** 2))
        got = _step_index_root_census(m, a, n1 ** 2, n2 ** 2, _K0_F)
        assert got == len(fiber_modes(m, a, n1 ** 2, n2 ** 2, _K0_F)), \
            (m, n1, n2, V, got)
    # (b) the SHORT list warns, naming both counts and the order
    n1, n2, V = 3.48, 1.44, 4.0
    a = V / (_K0_F * np.sqrt(n1 ** 2 - n2 ** 2))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gm = guided_modes(1, a, 6.0 * a, 400, n1 ** 2, n2 ** 2, _K0_F)
    n_exact = _step_index_root_census(1, a, n1 ** 2, n2 ** 2, _K0_F)
    assert n_exact == 3                       # the fixture is still short
    if len(gm) < n_exact:
        msgs = [str(w.message) for w in rec
                if "EXACT step-index hybrid" in str(w.message)]
        assert len(msgs) == 1, [str(w.message) for w in rec]
        assert f"returned {len(gm)} mode(s) for m = 1" in msgs[0], msgs[0]
        assert f"{n_exact} root(s)" in msgs[0], msgs[0]
    # (c) ... and a call whose census AGREES stays quiet
    a24 = 2.4 / (_K0_F * np.sqrt(1.45 ** 2 - 1.44 ** 2))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gm24 = guided_modes(1, a24, 6.0 * a24, 150, 1.45 ** 2, 1.44 ** 2,
                            _K0_F)
    assert len(gm24) == _step_index_root_census(
        1, a24, 1.45 ** 2, 1.44 ** 2, _K0_F) == 1
    assert not [w for w in rec
                if "EXACT step-index hybrid" in str(w.message)], \
        [str(w.message) for w in rec]
    # (d) the census refuses where it cannot be a census, instead of guessing
    assert _step_index_root_census(1, a, 3.48 ** 2 - 0.3j, 1.44 ** 2,
                                   _K0_F) is None          # lossy core
    assert _step_index_root_census(1, a, 1.44 ** 2, 3.48 ** 2,
                                   _K0_F) is None          # inverted profile
    # (e) census=False removes it
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        guided_modes(1, a, 6.0 * a, 200, n1 ** 2, n2 ** 2, _K0_F,
                     census=False)
    assert not [w for w in rec if "EXACT step-index hybrid" in str(w.message)]


# ===========================================================================
# H3 -- the validated-scope notice must reach the OUT-OF-PLANE path too
# ===========================================================================
def _offplane(cell, tilt=0.2):
    t = _promote(cell)
    t[..., 0, 2] = tilt * (cell - cell.flat[0])
    t[..., 2, 0] = tilt * (cell - cell.flat[0])
    return t


def test_h3_offplane_fff_nv_gets_the_scope_notice_too():
    """H3 (P2), second half -- the door the in-plane fix left open
    (VERIFY-A14 V6).  ``_li_tensor_scope_notice`` was called only from the
    IN-PLANE branch of ``rcwa_jones_2d``, so an OUT-OF-PLANE (full 3x3) tensor
    cell with a curved pattern got ``formulation='fff_nv'`` -- the same Li-2003
    staircase factorization, and per the WP's deferred D3 the UN-symmetrized
    one -- with no scope signal at all, while the identical in-plane cell
    warned.

    DECISION assertions (no numeric bar): an out-of-plane DISK warns exactly
    once and names the diagonal-boundary fraction; an out-of-plane SQUARE does
    not; ``laurent`` / ``li`` never do (they are rigorous for curved patterns);
    and ``allow_nonseparable_nv=True`` silences it.  The square arm is what
    makes this two-sided -- a notice that fired on every out-of-plane cell
    would pass a disk-only test.  MEASURED before: 0 notices on the
    out-of-plane disk.
    """
    S = _H4_CELL.shape[0]
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    disk = np.full((S, S), 2.25 + 0j)
    disk[X ** 2 + Y ** 2 <= 0.25 ** 2] = 6.25 + 0j

    def notices(cell, **kw):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            rcwa_jones_2d(0.5e-6, 0.5e-6, _offplane(cell), 1.0, 1.0, 0.3e-6,
                          0.633e-6, n_orders_x=3, n_orders_y=3, **kw)
        return [str(w.message) for w in rec if "NON-SEPARABLE" in str(w.message)]

    hit = notices(disk, formulation="fff_nv")
    assert len(hit) == 1, hit
    assert "of the cell boundary runs diagonal" in hit[0], hit[0]
    assert notices(_H4_CELL, formulation="fff_nv") == []          # square
    assert notices(disk, formulation="fff_nv",
                   allow_nonseparable_nv=True) == []
    for form in ("laurent", "li"):
        assert notices(disk, formulation=form) == [], form
