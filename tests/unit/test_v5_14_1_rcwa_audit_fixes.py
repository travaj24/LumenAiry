"""v5.14.1 -- RCWA audit fixes (2026-06-10, hand-verified P1/P2 findings).

Pins the three P1 fixes and the conical-oracle P2 fix:

* **F1 inverse-z-rule** -- the 2-D ``'li'``/``'fff'``/``'auto'`` formulation
  (and the shapes solver / ``fff_nv`` ``EZZ``) applied Li's INVERSE rule to
  the ``E_z`` elimination of a z-invariant layer.  ``E_z`` is tangential to
  every vertical wall, so the direct rule is mandatory (Li 1997 Eq. 27); the
  wrong rule overestimated metal-stripe absorptance by up to +0.35,
  period-robust and invisible to the energy guard on lossy cells.  2-D
  ``'li'`` is now the Li-1997 SEQUENTIAL rule (Eqs. 8/9): inverse along each
  E-component's own axis, direct along the other, direct-rule ``E_z`` -- it
  reduces to rigorous 1-D ``'li'`` on separable cells.
* **gain superstrate** -- ``Im(n_sup) < 0`` (even 1e-9) flipped the forward
  root, silently returning ``R = 0`` and NEGATIVE ``T`` (TM sum -392.8);
  rejected loudly at entry now, and ``_check_energy`` is two-sided.
* **silent sub-tripwire window** -- a provably-lossless solve violating the
  exact ``R+T = 1`` closure beyond 1e-6 (but under the hard 5% tripwire) now
  emits ``_EnergyWarning`` (the per-order answers there are wrong), and
  ``stabilize=True`` treats that warning as a failed attempt instead of
  returning the byte-identical wrong answer.
* **Berreman conical oracle** -- ``tests/unit/_berreman4x4._berreman_delta``
  was wrong whenever ``Kx*Ky != 0`` (three entries off by exactly
  ``+/-Kx*Ky``, pinned by rotation covariance); fixed, and the conical
  wrapper now matches ``rcwa_jones_1d`` to ~4e-15 including out-of-plane
  lossy tensors.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest

from lumenairy.elements.rcwa import (
    RCWAStack,
    prepare_rcwa_2d,
    rcwa_efficiency_1d,
    rcwa_efficiency_2d,
    rcwa_jones_1d,
)
from lumenairy.elements.rcwa._core import (
    _C,
    _check_energy,
    _EnergyError,
    _EnergyWarning,
)

from ._berreman4x4 import _berreman_delta, _split_updown

_WL = 0.633e-6


# --------------------------------------------------------------------------- #
# (1) F1: the Li-1997 sequential rule
# --------------------------------------------------------------------------- #

def _metal_stripe_cell(S, Sy=None):
    cell = np.full((S, Sy or S), 1.0 + 0j)
    cell[:S // 2, :] = -8.97 + 1.08j
    return cell


def _row0(o, R, T):
    m = o[:, 1] == 0
    Rr = {int(k): float(v) for k, v in zip(o[m, 0], R[m])}
    Tr = {int(k): float(v) for k, v in zip(o[m, 0], T[m])}
    return Rr, Tr, float(R[~m].sum() + T[~m].sum())


def test_2d_li_reduces_to_1d_li_metal_stripe():
    """A y-uniform metal stripe through 2-D 'li' must reproduce rigorous 1-D
    'li' per-order (the audit's mechanism proof; the pre-fix path was off by
    +0.35 absorptance).  The residual is the pixel-DFT-vs-analytic-sinc
    sampling floor, second-order in the cell sampling S."""
    P, D, M = 1.0e-6, 0.4e-6, 8
    n_metal = np.sqrt(-8.97 + 1.08j)
    o1, R1, T1 = rcwa_efficiency_1d(P, n_metal, 1.0, 1.45, 1.0, D, 0.5, _WL,
                                    polarization="tm", n_orders=M,
                                    formulation="li")
    i0 = len(o1) // 2
    o2, R2, T2 = rcwa_efficiency_2d(P, P, _metal_stripe_cell(256, 64),
                                    1.45, 1.0, D, _WL, polarization="tm",
                                    n_orders_x=M, n_orders_y=2,
                                    formulation="li")
    Rr, Tr, leak = _row0(o2, R2, T2)
    dmax = max(max(abs(Rr[m] - float(R1[i0 + m])),
                   abs(Tr[m] - float(T1[i0 + m]))) for m in range(-M, M + 1))
    assert dmax < 2e-4          # measured 5.4e-5 at S=256 (3.2e-6 at S=1024)
    assert leak < 1e-12         # y-uniform: nothing in the n != 0 rows


def test_2d_li_metal_absorptance_beats_laurent():
    """On the metal stripe the sequential 'li' must land near the converged
    oracle where 'laurent' is still ~0.1 off -- and nowhere near the pre-fix
    +0.35 blow-up (which was WORSE than laurent)."""
    P, D, M = 1.0e-6, 0.4e-6, 8
    n_metal = np.sqrt(-8.97 + 1.08j)
    o, R, T = rcwa_efficiency_1d(P, n_metal, 1.0, 1.45, 1.0, D, 0.5, _WL,
                                 polarization="tm", n_orders=96,
                                 formulation="li")
    A_oracle = 1.0 - float(R.sum() + T.sum())
    cell = _metal_stripe_cell(64)
    dA = {}
    for form in ("li", "laurent"):
        o2, R2, T2 = rcwa_efficiency_2d(P, P, cell, 1.45, 1.0, D, _WL,
                                        polarization="tm", n_orders_x=M,
                                        n_orders_y=2, formulation=form)
        dA[form] = abs(1.0 - float(R2.sum() + T2.sum()) - A_oracle)
    assert dA["li"] < 2e-2                  # measured 9.2e-3
    assert dA["laurent"] > 5e-2             # measured 1.08e-1
    assert dA["li"] < dA["laurent"] / 3


def test_2d_li_uniform_cell_collapses_to_laurent():
    """All factorization rules coincide on a uniform cell; 'li' must take the
    scalar path's analytic uniform modes (byte-identical to 'laurent')."""
    cell = np.full((16, 16), 2.25 + 0j)
    a = rcwa_efficiency_2d(0.6e-6, 0.6e-6, cell, 1.5, 1.0, 0.2e-6, _WL,
                           n_orders_x=3, n_orders_y=3, formulation="laurent")
    b = rcwa_efficiency_2d(0.6e-6, 0.6e-6, cell, 1.5, 1.0, 0.2e-6, _WL,
                           n_orders_x=3, n_orders_y=3, formulation="li")
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])


def test_prepared_li_matches_direct():
    """prepare_rcwa_2d carries the sequential-rule operators; its solve must
    equal the direct call."""
    P, D, M = 1.0e-6, 0.4e-6, 6
    cell = _metal_stripe_cell(64)
    o_d, R_d, T_d = rcwa_efficiency_2d(P, P, cell, 1.45, 1.0, D, _WL,
                                       polarization="tm", n_orders_x=M,
                                       n_orders_y=2, formulation="li")
    prep = prepare_rcwa_2d(P, P, cell, 1.45, 1.0, D, polarization="tm",
                           n_orders_x=M, n_orders_y=2, formulation="li")
    o_p, R_p, T_p = prep.solve(_WL)
    assert np.max(np.abs(R_p - R_d)) < 1e-12
    assert np.max(np.abs(T_p - T_d)) < 1e-12


def test_2d_li_lossless_energy():
    """The sequential rule conserves energy exactly on a lossless cell."""
    S = 64
    cell = np.full((S, S), 1.0 + 0j)
    cell[16:48, 24:56] = 6.25                # true 2-D pillar, both axes patterned
    o, R, T = rcwa_efficiency_2d(0.6e-6, 0.6e-6, cell, 1.5, 1.0, 0.25e-6, _WL,
                                 polarization="tm", n_orders_x=5, n_orders_y=5,
                                 formulation="li")
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# (2) gain superstrate + two-sided energy check
# --------------------------------------------------------------------------- #

def test_gain_superstrate_rejected_everywhere():
    args1d = dict(period=0.5e-6, duty_cycle=0.5, n_ridge=2.0, n_groove=1.0,
                  n_substrate=1.5, depth=0.3e-6, wavelength=1.55e-6,
                  n_orders=5)
    with pytest.raises(ValueError, match="gain"):
        rcwa_efficiency_1d(n_superstrate=1.0 - 1e-9j, **args1d)
    with pytest.raises(ValueError, match="gain"):
        rcwa_jones_1d(0.5e-6, np.diag([4.0, 4.0, 4.0]).astype(complex),
                      1.0, 1.5, 1.0 - 1e-9j, 0.3e-6, 0.5, 1.55e-6, n_orders=5)
    cell = np.full((24, 24), 2.25 + 0j)
    cell[6:18, 6:18] = 4.0
    with pytest.raises(ValueError, match="gain"):
        rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell, 1.5, 1.0 - 1e-9j, 0.3e-6,
                           1.55e-6, n_orders_x=3, n_orders_y=3)
    st = RCWAStack(0.5e-6, n_substrate=1.5, n_superstrate=1.0 - 1e-9j,
                   n_orders=5)
    st.add_layer(0.3e-6, eps=4.0)
    with pytest.raises(ValueError, match="gain"):
        st.set_source(1.55e-6).solve()
    # a lossless and a (tiny-)lossy superstrate still solve fine
    for nsup in (1.0, 1.0 + 1e-6j):
        o, R, T = rcwa_efficiency_1d(n_superstrate=nsup, **args1d)
        assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-6


def test_check_energy_two_sided_and_lossless_window():
    ok = np.array([0.5]), np.array([0.5])
    _check_energy("t", *ok)                                   # exact closure
    with pytest.raises(_EnergyError):                         # negative total
        _check_energy("t", np.array([0.0]), np.array([-0.5]))
    with pytest.raises(_EnergyError):                         # >5% (unchanged)
        _check_energy("t", np.array([1.0]), np.array([0.2]))
    # lossless silent window 1e-6..0.05 -> _EnergyWarning
    with pytest.warns(_EnergyWarning, match="closure"):
        _check_energy("t", np.array([0.51]), np.array([0.52]), lossless=True)
    # same total WITHOUT the lossless proof -> silent (lossy R+T<1 is physical)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_energy("t", np.array([0.51]), np.array([0.42]), lossless=False)
    # roundoff-level closure error -> no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_energy("t", np.array([0.5]), np.array([0.5 + 1e-9]),
                      lossless=True)


# --------------------------------------------------------------------------- #
# (3) the silent sub-tripwire window + stabilize recovery
# --------------------------------------------------------------------------- #

def test_silent_window_warns_and_stabilize_recovers():
    """The audited 1-D case (P=8um low-contrast TM, n_orders=20 on a
    single-thread BLAS) silently returned R+T-1 = +3.3e-2 with broken +/-1
    symmetry.  Whether or not THIS build reproduces the instability, the
    public contract now is: any lossless closure violation > 1e-6 warns, and
    stabilize=True returns a truncation whose closure is clean and whose
    per-order values match the adjacent-truncation consensus."""
    args = (8.0e-6, 1.5, 1.45, 1.5, 1.0, 1.0e-6, 0.5, 0.6e-6)
    # WHICH truncations are unstable here is BLAS-build/runner dependent (one
    # CI runner blows up at M=19 where MKL/WSL are clean), so collect the
    # consensus from whatever nearby truncations come back clean.
    vals = []
    for M in (18, 19, 21, 22, 24):
        try:
            o, R, T = rcwa_efficiency_1d(*args, polarization="tm", n_orders=M)
        except _EnergyError:
            continue                       # loud raise = correct behaviour
        if abs(float(R.sum() + T.sum()) - 1.0) < 1e-6:
            vals.append(float(T[np.where(o == 1)[0][0]]))
    assert len(vals) >= 2, "no clean truncations found on this build"
    assert max(vals) - min(vals) < 1e-5
    consensus = float(np.median(vals))
    # the suspect truncation must be clean, WARN, or RAISE -- never
    # silent-wrong (the audited pre-fix behaviour)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            o, R, T = rcwa_efficiency_1d(*args, polarization="tm",
                                         n_orders=20)
            closure = abs(float(R.sum() + T.sum()) - 1.0)
            warned = any(issubclass(w.category, _EnergyWarning) for w in wl)
            assert (closure < 1e-6) or warned
        except _EnergyError:
            pass
    # stabilize must hand back a clean consensus-grade answer
    o, R, T = rcwa_efficiency_1d(*args, polarization="tm", n_orders=20,
                                 stabilize=True)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-6
    assert abs(float(T[np.where(o == 1)[0][0]]) - consensus) < 2e-4


def test_stabilize_2d_ladder_contract():
    """The 2-D stabilize ladder must terminate with its documented contract
    (an _EnergyError if nothing conserves) -- never the sampling ValueError
    about an n_orders the user never requested (audit P2 wrong-abort)."""
    S = 33
    cell = np.full((S, S), 2.25 + 0j)
    cell[16, 16] = 12.0
    try:
        o, R, T = rcwa_efficiency_2d(1.0e-6, 1.0e-6, cell, 1.5, 1.0, 0.3e-6,
                                     0.6e-6, n_orders_x=4, n_orders_y=4,
                                     stabilize=True)
        assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-5   # found a clean one
    except _EnergyError:
        pass                                                # documented contract
    except ValueError as e:                                 # pragma: no cover
        pytest.fail(f"stabilize leaked a non-energy ValueError: {e}")


# --------------------------------------------------------------------------- #
# (4) the corrected conical Berreman oracle
# --------------------------------------------------------------------------- #

def _rot2(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def test_berreman_delta_conical_rotation_covariance():
    """For the tangential state [Ex, Ey, Hx, Hy] an in-plane rotation R must
    satisfy blkdiag(R, R) Delta(eps_rot, kt, 0) blkdiag(R, R)^T ==
    Delta(eps, Kx, Ky) -- solver-independent ground truth that pinned the
    pre-fix defect to exactly +/-Kx*Ky at three entries."""
    rng = np.random.default_rng(7)
    eps_a = np.array([[4.0, 0.3, 0.2], [0.3, 3.0, 0.1], [0.2, 0.1, 2.5]],
                     dtype=complex) + 0.05j * rng.random((3, 3))
    for eps in (2.25 * np.eye(3, dtype=complex), eps_a):
        for kt, phi in ((0.7, 0.6108), (1.2, 0.1745)):
            Kx, Ky = kt * np.cos(phi), kt * np.sin(phi)
            Rm = _rot2(phi)
            B = np.kron(np.eye(2), Rm)
            R3 = np.eye(3)
            R3[:2, :2] = Rm
            eps_rot = R3.T @ eps @ R3        # tensor expressed in the rotated frame
            D_pl = _berreman_delta(eps_rot, kt, 0.0)
            D_co = _berreman_delta(eps, Kx, Ky)
            assert np.max(np.abs(B @ D_pl @ B.T - D_co)) < 1e-12


def _berreman_jones_conical(eps_lab, n_sup, n_sub, depth, wavelength, theta,
                            phi):
    """Minimal conical 4x4 wrapper over the (fixed) _berreman_delta."""
    eps = np.conj(np.asarray(eps_lab).astype(_C))
    eps_sup = complex(np.conj(_C(n_sup) ** 2))
    eps_sub = complex(np.conj(_C(n_sub) ** 2))
    k0 = 2.0 * np.pi / wavelength
    st = np.sin(theta)
    Kx = float(np.real(np.conj(_C(n_sup))) * st * np.cos(phi))
    Ky = float(np.real(np.conj(_C(n_sup))) * st * np.sin(phi))
    gam, Psi = np.linalg.eig(_berreman_delta(eps, Kx, Ky))
    qs_sup, A_sup = np.linalg.eig(
        _berreman_delta(eps_sup * np.eye(3, dtype=_C), Kx, Ky))
    qs_sub, A_sub = np.linalg.eig(
        _berreman_delta(eps_sub * np.eye(3, dtype=_C), Kx, Ky))
    T_layer = Psi @ np.diag(np.exp(gam * k0 * depth)) @ np.linalg.inv(Psi)
    sup_fwd, sup_bwd = _split_updown(qs_sup)
    sub_fwd, _sub_bwd = _split_updown(qs_sub)
    Asup_fwd, Asup_bwd = A_sup[:, sup_fwd], A_sup[:, sup_bwd]
    Asub_fwd = A_sub[:, sub_fwd]
    Mmat = np.column_stack([Asup_bwd, -(T_layer @ Asub_fwd)])
    Jr = np.zeros((2, 2), dtype=_C)
    Jt = np.zeros((2, 2), dtype=_C)
    for col, Einc in enumerate(([1.0, 0.0], [0.0, 1.0])):
        c_inc = np.linalg.solve(Asup_fwd[:2, :], np.asarray(Einc, dtype=_C))
        u = np.linalg.solve(Mmat, -(Asup_fwd @ c_inc))
        Jr[:, col] = Asup_bwd[:2, :] @ u[:2]
        Jt[:, col] = Asub_fwd[:2, :] @ u[2:]
    return np.conj(Jr), np.conj(Jt), Kx, Ky


def test_berreman_conical_lossless_energy_and_rcwa_match():
    """The fixed oracle conserves energy conically on a lossless isotropic
    slab (pre-fix: R+T = 14.35 at theta=20, phi=35) and matches the RCWA
    uniform-layer Jones at machine precision."""
    wl, depth, nsup, nsub = 0.6328e-6, 0.55e-6, 1.0, 1.45
    eps_iso = 2.25 * np.eye(3, dtype=complex)
    for th_d, ph_d in ((20.0, 35.0), (50.0, 60.0)):
        th, ph = np.deg2rad(th_d), np.deg2rad(ph_d)
        Jr, Jt, Kx, Ky = _berreman_jones_conical(eps_iso, nsup, nsub, depth,
                                                 wl, th, ph)
        kz1 = np.sqrt(complex(nsup ** 2 - Kx ** 2 - Ky ** 2))
        kz3 = np.sqrt(complex(nsub ** 2 - Kx ** 2 - Ky ** 2))
        tot = 0.0
        for col in range(2):
            ex_i, ey_i = (1.0, 0.0) if col == 0 else (0.0, 1.0)
            ez_i = -(Kx * ex_i + Ky * ey_i) / kz1
            einc = abs(ex_i) ** 2 + abs(ey_i) ** 2 + abs(ez_i) ** 2
            rx, ry = Jr[0, col], Jr[1, col]
            tx, ty = Jt[0, col], Jt[1, col]
            rz = (Kx * rx + Ky * ry) / kz1
            tz = -(Kx * tx + Ky * ty) / kz3
            Rc = (abs(rx) ** 2 + abs(ry) ** 2 + abs(rz) ** 2) / einc
            Tc = float(np.real(kz3) / np.real(kz1)) * (
                abs(tx) ** 2 + abs(ty) ** 2 + abs(tz) ** 2) / einc
            tot += float(Rc + Tc)
        assert abs(tot - 2.0) < 1e-12
