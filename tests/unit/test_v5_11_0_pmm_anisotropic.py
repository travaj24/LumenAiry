"""v5.11.0 (shipped 5.10.7) anisotropic PMM -- ``pmm_jones_1d``.

The spectral-element counterpart of the FMM tensor path ``rcwa_jones_1d``: a
binary 1-D grating whose ridge / groove are full ``(3, 3)`` in-plane permittivity
tensors, returning the full complex ``2x2`` Jones reflection.  The off-diagonal
``exy`` couples ``E_x`` <-> ``E_y`` in the coupled spectral-element modal
eigenproblem.

Validation gates (STEP-0 of the build, promoted to regression tests):
  1. vs ``rcwa_jones_1d`` (the oracle): the 2x2 Jones must MATCH, and converge
     spectrally in degree with no plateau -- a lossless tilted-LC grating AND a
     lossy anisotropic metal case.
  2. DECOUPLE: a diagonal tensor (``exy = eyx = 0``) reduces to the scalar
     ``pmm_efficiency_1d`` (TE from ``eyy``, TM from ``exx``); cross-pol ~0.
  3. ENERGY: lossless -> ``sum(R)+sum(T) = 1`` per incident polarization (cross-
     pol included).
  4. NO FLOOR: the clean-degree Jones error vs the oracle decreases with degree.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    pmm_1d,
    pmm_efficiency_1d,
    pmm_jones_1d,
    pmm_jones_1d_segments,
    pmm_jones_1d_slanted,
    pmm_jones_1d_slanted_segments,
)
from lumenairy.elements.rcwa import rcwa_efficiency_1d, rcwa_jones_1d, uniaxial_tensor

_C = np.complex128

# lossless tilted-LC binary grating (the canonical anisotropic case): an in-plane
# director (theta=pi/2), period ~1um, wl ~1.5um, normal incidence, only the 0
# order propagates (period/wl < 1).
LC = dict(period=1.0e-6, depth=0.5e-6, duty_cycle=0.5, wavelength=1.5e-6,
          n_substrate=1.0, n_superstrate=1.0)
LC_RIDGE = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=0.4)
LC_GROOVE = (1.0 ** 2) * np.eye(3, dtype=_C)

# lossy anisotropic "metal-like" ridge (birefringent + absorbing)
MET = dict(period=1.0e-6, depth=0.2e-6, duty_cycle=0.5, wavelength=1.5e-6,
           n_substrate=1.0, n_superstrate=1.0)
MET_RIDGE = uniaxial_tensor(2.0 + 0.3j, 2.5 + 0.5j, np.pi / 2, phi=0.4)
MET_GROOVE = (1.0 ** 2) * np.eye(3, dtype=_C)


# --------------------------------------------------------------------------- #
# GATE 1 -- vs the rcwa_jones_1d oracle (the 2x2 Jones matches)
# --------------------------------------------------------------------------- #
def test_jones_matches_rcwa_oracle_lossless_lc():
    o, R, T, J = pmm_jones_1d(LC["period"], LC_RIDGE, LC_GROOVE,
                              LC["n_substrate"], LC["n_superstrate"],
                              LC["depth"], LC["duty_cycle"], LC["wavelength"],
                              degree=24)
    oo, Ro, To, Jo = rcwa_jones_1d(LC["period"], LC_RIDGE, LC_GROOVE,
                                   LC["n_substrate"], LC["n_superstrate"],
                                   LC["depth"], LC["duty_cycle"],
                                   LC["wavelength"], n_orders=60)
    assert J.shape == (2, 2)
    assert np.max(np.abs(J - Jo)) < 1e-3        # full complex Jones, both pols


def test_jones_matches_rcwa_oracle_lossy_metal():
    o, R, T, J = pmm_jones_1d(MET["period"], MET_RIDGE, MET_GROOVE,
                              MET["n_substrate"], MET["n_superstrate"],
                              MET["depth"], MET["duty_cycle"], MET["wavelength"],
                              degree=24)
    oo, Ro, To, Jo = rcwa_jones_1d(MET["period"], MET_RIDGE, MET_GROOVE,
                                   MET["n_substrate"], MET["n_superstrate"],
                                   MET["depth"], MET["duty_cycle"],
                                   MET["wavelength"], n_orders=80)
    assert np.max(np.abs(J - Jo)) < 1e-3


def test_jones_offdiagonal_is_nonzero():
    # the whole point: an in-plane tilted director produces genuine cross-pol
    o, R, T, J = pmm_jones_1d(LC["period"], LC_RIDGE, LC_GROOVE,
                              LC["n_substrate"], LC["n_superstrate"],
                              LC["depth"], LC["duty_cycle"], LC["wavelength"],
                              degree=24)
    assert abs(J[0, 1]) > 1e-3 and abs(J[1, 0]) > 1e-3


# --------------------------------------------------------------------------- #
# GATE 2 -- DECOUPLE: a diagonal tensor reduces to the scalar PMM
# --------------------------------------------------------------------------- #
DEC = dict(period=1.0e-6, depth=0.3e-6, duty_cycle=0.5, wavelength=0.8e-6,
           n_substrate=1.0, n_superstrate=1.0)
_NR, _NG = 2.5, 1.0


def test_decouple_cross_pol_vanishes():
    er = (_NR ** 2) * np.eye(3, dtype=_C)
    eg = (_NG ** 2) * np.eye(3, dtype=_C)
    o, R, T, J = pmm_jones_1d(DEC["period"], er, eg, DEC["n_substrate"],
                              DEC["n_superstrate"], DEC["depth"],
                              DEC["duty_cycle"], DEC["wavelength"], degree=20)
    # diagonal tensor -> no E_x<->E_y coupling -> Jones is diagonal
    assert abs(J[0, 1]) < 1e-9
    assert abs(J[1, 0]) < 1e-9


def test_decouple_reduces_to_scalar_efficiencies():
    # the decoupled co-pol channels must match the scalar PMM/FMM truth:
    #   incident E_y == TE (E along the grooves);  incident E_x == TM
    er = (_NR ** 2) * np.eye(3, dtype=_C)
    eg = (_NG ** 2) * np.eye(3, dtype=_C)
    o, R, T, J = pmm_jones_1d(DEC["period"], er, eg, DEC["n_substrate"],
                              DEC["n_superstrate"], DEC["depth"],
                              DEC["duty_cycle"], DEC["wavelength"], degree=20)
    # FMM at very high order = ground truth both must reduce to
    _, Rte, Tte = rcwa_efficiency_1d(
        DEC["period"], _NR, _NG, DEC["n_substrate"], DEC["n_superstrate"],
        DEC["depth"], DEC["duty_cycle"], DEC["wavelength"],
        polarization="te", n_orders=120)
    _, Rtm, Ttm = rcwa_efficiency_1d(
        DEC["period"], _NR, _NG, DEC["n_substrate"], DEC["n_superstrate"],
        DEC["depth"], DEC["duty_cycle"], DEC["wavelength"],
        polarization="tm", n_orders=120)
    # incident E_y row == TE; incident E_x row == TM
    assert abs(float(R[1].sum()) - float(Rte.sum())) < 1e-4   # TE reflection
    assert abs(float(T[1].sum()) - float(Tte.sum())) < 1e-4
    assert abs(float(R[0].sum()) - float(Rtm.sum())) < 2e-3   # TM (slower rate)
    assert abs(float(T[0].sum()) - float(Ttm.sum())) < 2e-3


def test_decouple_jones_matches_scalar_pmm_reflection_amplitude():
    # the diagonal Jones entry magnitudes equal the scalar PMM zeroth-order
    # reflectance (|r|^2) -- sanity that the coupled amplitude IS the scalar one
    er = (_NR ** 2) * np.eye(3, dtype=_C)
    eg = (_NG ** 2) * np.eye(3, dtype=_C)
    o, R, T, J = pmm_jones_1d(DEC["period"], er, eg, DEC["n_substrate"],
                              DEC["n_superstrate"], DEC["depth"],
                              DEC["duty_cycle"], DEC["wavelength"], degree=20)
    o_te, R_te, T_te = pmm_efficiency_1d(
        DEC["period"], _NR, _NG, DEC["n_substrate"], DEC["n_superstrate"],
        DEC["depth"], DEC["duty_cycle"], DEC["wavelength"],
        polarization="te", degree=20)
    m0 = np.where(o_te == 0)[0][0]
    # |J_yy|^2 (incident Ey, reflected Ey, order 0) == scalar TE R order 0
    assert abs(abs(J[1, 1]) ** 2 - float(R_te[m0])) < 5e-3


# --------------------------------------------------------------------------- #
# GATE 3 -- energy conservation (lossless), cross-pol included
# --------------------------------------------------------------------------- #
def test_lossless_conserves_energy_both_polarizations():
    o, R, T, J = pmm_jones_1d(LC["period"], LC_RIDGE, LC_GROOVE,
                              LC["n_substrate"], LC["n_superstrate"],
                              LC["depth"], LC["duty_cycle"], LC["wavelength"],
                              degree=24)
    for row in range(2):                        # incident Ex, incident Ey
        tot = float(R[row].sum() + T[row].sum())
        assert abs(tot - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# GATE 4 -- NO FLOOR: the clean-degree Jones error improves with degree
# --------------------------------------------------------------------------- #
def test_no_accuracy_floor_jones_error_improves():
    oo, Ro, To, Jo = rcwa_jones_1d(LC["period"], LC_RIDGE, LC_GROOVE,
                                   LC["n_substrate"], LC["n_superstrate"],
                                   LC["depth"], LC["duty_cycle"],
                                   LC["wavelength"], n_orders=80)

    def clean_err(deg):
        # stabilize=False probes the EXACT degree; a resonant degree INFLATES the
        # power and returns None.  Which degrees resonate is LAPACK-build
        # dependent (the same isolated-resonance pathology as the scalar PMM), so
        # the no-floor property is the running-min over CLEAN degrees, not a pair
        # of fixed degrees (degree 20 happened to resonate on the CI build).
        o, R, T, J = pmm_jones_1d(
            LC["period"], LC_RIDGE, LC_GROOVE, LC["n_substrate"],
            LC["n_superstrate"], LC["depth"], LC["duty_cycle"],
            LC["wavelength"], degree=deg, stabilize=False)
        tot = float(R.sum() + T.sum())
        passive = tot <= 2.0 + 2e-3
        return (float(np.max(np.abs(J - Jo))) if passive else None)

    errs = {d: clean_err(d) for d in range(8, 25)}
    low = [e for d, e in errs.items() if d <= 12 and e is not None]
    high = [e for d, e in errs.items() if d >= 16 and e is not None]
    assert low and high                          # clean degrees exist at both ends
    assert min(high) < min(low)                  # running-min improves, no floor
    assert min(high) < 1e-4                       # reaches deep accuracy


# --------------------------------------------------------------------------- #
# the scalar pmm_efficiency_1d is unchanged by the new path (bit-identical)
# --------------------------------------------------------------------------- #
def test_scalar_pmm_unchanged_by_jones_addition():
    # the new anisotropic path is separate; the scalar solver must be unperturbed.
    # Anchor to the FMM oracle (the v5.8 scalar contract): TE lossy-metal grating.
    gold = dict(period=1.0e-6, n_ridge=0.2 + 6j, n_groove=1.0, n_substrate=1.0,
                n_superstrate=1.0, depth=0.3e-6, duty_cycle=0.5,
                wavelength=0.8e-6)
    o, R, T = pmm_efficiency_1d(**gold, polarization="te", degree=16)
    of, Rf, Tf = rcwa_efficiency_1d(**gold, polarization="te", n_orders=80)
    assert abs(float(T.sum()) - float(Tf.sum())) < 5e-5     # scalar PMM == oracle
    # passive (lossy metal absorbs): R+T < 1, never inflated
    assert float(R.sum() + T.sum()) < 1.0 + 1e-6


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_oblique_jones_matches_fmm():
    # Oblique anisotropic Jones (the +i*kx0 Bloch shift, the Poynting-flux mode
    # selector, the per-column p-pol incident normalization) -- once a
    # NotImplementedError, now solved: the 2x2 Jones matches rcwa_jones_1d and
    # energy is conserved (lossless LC) at off-normal incidence.
    for ang_deg in (15.0, 30.0, 45.0):
        ang = np.radians(ang_deg)
        o, R, T, J = pmm_jones_1d(
            LC["period"], LC_RIDGE, LC_GROOVE, LC["n_substrate"],
            LC["n_superstrate"], LC["depth"], LC["duty_cycle"],
            LC["wavelength"], angle=ang, degree=20)
        oo, Ro, To, Jo = rcwa_jones_1d(
            LC["period"], LC_RIDGE, LC_GROOVE, LC["n_substrate"],
            LC["n_superstrate"], LC["depth"], LC["duty_cycle"],
            LC["wavelength"], angle=ang, n_orders=25)
        assert np.max(np.abs(J - Jo)) < 5e-4, f"ang={ang_deg}"
        # lossless -> both incident polarizations conserve energy
        assert np.allclose(R.sum(axis=1) + T.sum(axis=1), 1.0, atol=1e-6)


def _ovl_perorder(o1, R1, T1, o2, R2, T2):
    c = np.intersect1d(o1, o2)
    i1 = {int(x): k for k, x in enumerate(o1)}
    i2 = {int(x): k for k, x in enumerate(o2)}
    return max(max(np.max(np.abs(R1[:, i1[int(m)]] - R2[:, i2[int(m)]])),
                   np.max(np.abs(T1[:, i1[int(m)]] - T2[:, i2[int(m)]]))) for m in c)


@pytest.mark.parametrize("ang_deg", [0.0, 20.0])
def test_out_of_plane_matches_rcwa(ang_deg):
    """A tilted-director (theta=pi/4) tensor has OUT-OF-PLANE coupling
    (eps_xz/eps_yz != 0) -- now SUPPORTED via the native full-3x3 metric generator
    (was a ValueError).  pmm_jones_1d matches the rigorous rcwa_jones_1d oracle
    per-order to <1.5e-3 and conserves energy, at normal AND oblique incidence, on
    a multi-order grating (period > wl)."""
    op = uniaxial_tensor(1.5, 1.7, np.pi / 4, phi=0.3)
    gr = np.eye(3, dtype=_C)
    geom = dict(period=1.0e-6, depth=0.5e-6, duty_cycle=0.5, wavelength=0.633e-6,
                n_substrate=1.0, n_superstrate=1.0)
    a = np.deg2rad(ang_deg)
    oP, RP, TP, JP = pmm_jones_1d(
        geom["period"], op, gr, geom["n_substrate"], geom["n_superstrate"],
        geom["depth"], geom["duty_cycle"], geom["wavelength"], angle=a,
        degree=16, elements_per_region=4)
    oR, RR, TR, JR = rcwa_jones_1d(
        geom["period"], op, gr, geom["n_substrate"], geom["n_superstrate"],
        geom["depth"], geom["duty_cycle"], geom["wavelength"], angle=a,
        n_orders=61)
    assert _ovl_perorder(oP, RP, TP, oR, RR, TR) < 1.5e-3
    assert abs(RP[0].sum() + TP[0].sum() - 1.0) < 1e-3
    assert abs(RP[1].sum() + TP[1].sum() - 1.0) < 1e-3


def test_out_of_plane_in_plane_path_unchanged():
    """The out-of-plane route does NOT perturb the in-plane path: an IN-PLANE
    tensor (theta=pi/2 director, no out-of-plane coupling) still takes the
    second-order _pmm_jones_solve and matches rcwa_jones_1d as before."""
    ip = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=0.4)        # in-plane (eps_xz=0)
    gr = np.eye(3, dtype=_C)
    oP, RP, TP, JP = pmm_jones_1d(1.0e-6, ip, gr, 1.0, 1.0, 0.5e-6, 0.5, 0.633e-6,
                                  degree=16)
    oR, RR, TR, JR = rcwa_jones_1d(1.0e-6, ip, gr, 1.0, 1.0, 0.5e-6, 0.5, 0.633e-6,
                                   n_orders=61)
    assert _ovl_perorder(oP, RP, TP, oR, RR, TR) < 1.5e-3
    assert np.max(np.abs(JP - JR)) < 1e-3


def test_exported_top_level():
    import lumenairy as la
    assert hasattr(la, "pmm_jones_1d")


# ---------------------------------------------------------------------------
# Unified pmm_1d dispatcher
# ---------------------------------------------------------------------------
def _jeq(a, b):
    return (np.array_equal(a[0], b[0]) and np.max(np.abs(a[1] - b[1])) < 1e-14
            and np.max(np.abs(a[2] - b[2])) < 1e-14
            and np.max(np.abs(a[3] - b[3])) < 1e-14)


def test_pmm_1d_dispatch_bit_identical():
    """pmm_1d auto-routes to the right solver by geometry, BIT-IDENTICALLY."""
    er = np.diag([2.25, 2.10, 2.25]).astype(_C)
    er[0, 1] = er[1, 0] = 0.15
    eg = np.eye(3, dtype=_C)
    g = dict(period=1.0e-6, n_substrate=1.5, n_superstrate=1.0, depth=0.5e-6,
             wavelength=0.633e-6)
    phi = np.deg2rad(30.0)
    segs = [(0.5, er), (0.5, eg)]
    # binary vertical
    assert _jeq(
        pmm_1d(**g, eps_ridge=er, eps_groove=eg, duty_cycle=0.5, degree=16,
               stabilize=False),
        pmm_jones_1d(1.0e-6, er, eg, 1.5, 1.0, 0.5e-6, 0.5, 0.633e-6, degree=16,
                     stabilize=False))
    # binary slanted
    assert _jeq(
        pmm_1d(**g, eps_ridge=er, eps_groove=eg, slant_angle=phi, degree=16,
               stabilize=False),
        pmm_jones_1d_slanted(1.0e-6, er, eg, 1.5, 1.0, 0.5e-6, 0.5, 0.633e-6, phi,
                             degree=16, stabilize=False))
    # multi-region vertical
    assert _jeq(
        pmm_1d(**g, segments=segs, degree=16, stabilize=False),
        pmm_jones_1d_segments(1.0e-6, segs, 1.5, 1.0, 0.5e-6, 0.633e-6, degree=16,
                              stabilize=False))
    # multi-region slanted
    assert _jeq(
        pmm_1d(**g, segments=segs, slant_angle=phi, degree=16, stabilize=False),
        pmm_jones_1d_slanted_segments(1.0e-6, segs, 1.5, 1.0, 0.5e-6, 0.633e-6,
                                      phi, degree=16, stabilize=False))


def test_pmm_1d_out_of_plane_and_errors():
    """pmm_1d carries out-of-plane (vertical) through, and validates the spec."""
    op = np.diag([2.25, 2.10, 2.40]).astype(_C)
    op[0, 2] = op[2, 0] = 0.3
    g = dict(period=1.0e-6, n_substrate=1.5, n_superstrate=1.0, depth=0.5e-6,
             wavelength=0.633e-6)
    o, R, T, J = pmm_1d(**g, eps_ridge=op, eps_groove=np.eye(3, dtype=_C),
                        degree=14, elements_per_region=6)
    assert abs(R[0].sum() + T[0].sum() - 1.0) < 1e-3
    with pytest.raises(ValueError):                  # neither spec
        pmm_1d(**g)
    with pytest.raises(ValueError):                  # both specs
        pmm_1d(**g, eps_ridge=op, eps_groove=np.eye(3, dtype=_C),
               segments=[(1.0, op)])
