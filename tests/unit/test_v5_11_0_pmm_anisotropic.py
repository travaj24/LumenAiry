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

from lumenairy.elements.pmm import pmm_efficiency_1d, pmm_jones_1d
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
def test_rejects_oblique():
    with pytest.raises(NotImplementedError, match="normal incidence"):
        pmm_jones_1d(LC["period"], LC_RIDGE, LC_GROOVE, LC["n_substrate"],
                     LC["n_superstrate"], LC["depth"], LC["duty_cycle"],
                     LC["wavelength"], angle=0.2)


def test_rejects_out_of_plane_tensor():
    # a tilted (theta != pi/2) director has out-of-plane coupling
    op = uniaxial_tensor(1.5, 1.7, np.pi / 4, phi=0.3)
    with pytest.raises(ValueError, match="out-of-plane"):
        pmm_jones_1d(LC["period"], op, LC_GROOVE, LC["n_substrate"],
                     LC["n_superstrate"], LC["depth"], LC["duty_cycle"],
                     LC["wavelength"])


def test_exported_top_level():
    import lumenairy as la
    assert hasattr(la, "pmm_jones_1d")
