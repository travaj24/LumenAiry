"""AUDIT_V5_24_2 S5-5 regression: the engine->propagator field bridge is
engine-agnostic.

Before S5-5 the per-order carrier + power-normalization math lived ONLY on
``RCWAResult.to_multiorder_field`` / ``.to_jones_field``; PMM and 2-D PMM
exposed raw ``2x2`` Jones and raw ``per_order_amplitudes`` but no supported way
to turn that modal data into a propagatable field (a strongly-diffracting cell
spreads most power into non-zero orders the raw Jones drops).  The fix factors
the math into the free ``jones_field_from_orders(amps, ...)`` keyed on the
shared ``per_order_amplitudes`` dict contract -- now additionally carrying the
incidence terms ``kz_inc`` / ``kx0`` / ``ky0`` that the transmission-port flux
weight ``Re(kz_m/kz_inc)`` needs (the transmission dict's ``kz`` is the
SUBSTRATE kz, not the incident-medium one) -- and delegates
``RCWAResult.to_multiorder_field`` to it (byte-identical).

Independent oracle
------------------
On a grid tiling exactly one unit cell the propagating-order plane-wave
carriers are orthonormal, so Parseval makes ``mean|field|^2`` equal the SUM of
the solver's own per-order diffraction EFFICIENCIES -- a quantity the field
bridge never touches (the solver builds it from the Poynting flux bookkeeping,
the bridge from the tangential amplitudes + the reconstructed carriers).  The
bridge closing that identity for BOTH RCWA and PMM (independent solvers that
agree on the amplitudes only to ~1e-3) validates the augmented dict + the shared
carrier / power math end to end -- not a tautology against the code under test.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.polarization import jones_field_from_orders
from lumenairy.elements.rcwa import RCWAStack

_WL = 1.55e-6
_PERIOD = 1.1e-6
_DUTY, _DEPTH = 0.45, 0.35e-6
_EPS_R, _EPS_G = 4.2 + 0j, 1.0 + 0j
_NSUB = 1.45


def _pmm_grating(theta, phi=0.0):
    s = PMMStack(_PERIOD, n_substrate=_NSUB, degree=14, far_field_orders=11)
    s.add_layer(_DEPTH, segments=[(_DUTY, _EPS_R), (1 - _DUTY, _EPS_G)])
    s.set_source(_WL, theta=theta, phi=phi)
    return s


def _rcwa_grating(theta, phi=0.0, n_orders=41, S=2048):
    s = RCWAStack(_PERIOD, n_substrate=_NSUB, n_orders=n_orders)
    x = (np.arange(S) + 0.5) / S
    cell = np.where(x < _DUTY, _EPS_R, _EPS_G)[:, None]
    s.add_layer(_DEPTH, eps_cell=cell, formulation="li")
    s.set_source(_WL, theta=theta, phi=phi)
    return s


def _eff_total(eff_row, kz):
    """Sum of the PROPAGATING-order efficiencies for one incident pol."""
    prop = np.real(np.asarray(kz)) > 1e-12
    return float(np.asarray(eff_row)[prop].sum())


def _field_power(field):
    return float(np.mean(np.abs(field.Ex) ** 2 + np.abs(field.Ey) ** 2))


# ======================================================================== #
# The gap S5-5 closes: PMM now has a supported field bridge.               #
# ======================================================================== #

@pytest.mark.parametrize("theta_deg", [0.0, 18.0])
@pytest.mark.parametrize("port", ["reflection", "transmission"])
def test_s5_5_pmm_bridge_parseval_energy(theta_deg, port):
    """jones_field_from_orders on a PMM per_order_amplitudes dict rebuilds an
    energy-correct field: on a one-cell grid its mean power equals the SUM of
    PMM's own propagating-order efficiencies (the independent oracle)."""
    sp = _pmm_grating(np.deg2rad(theta_deg))
    _o, R, T, _J = sp.solve()
    amps = sp.per_order_amplitudes(port)
    eff_row = (R if port == "reflection" else T)[0]        # incident Ex
    e_tot = _eff_total(eff_row, amps["kz"])
    nx = 60                                                 # > 2 * far orders
    jf = jones_field_from_orders(amps, nx, nx, _PERIOD / nx,
                                 incident=(1.0, 0.0), normalize="power")
    assert abs(_field_power(jf) - e_tot) < 1e-6


def test_s5_5_pmm_transmission_is_multiorder():
    """The transmission side of this grating genuinely spreads power across
    >= 2 propagating orders (so the Parseval gate above is not vacuous / a
    single specular carrier)."""
    sp = _pmm_grating(np.deg2rad(18.0))
    sp.solve()
    amps = sp.per_order_amplitudes("transmission")
    assert int(np.sum(np.real(amps["kz"]) > 1e-12)) >= 2


# ======================================================================== #
# RCWA delegation: same bridge, still energy-correct, byte-identical path. #
# ======================================================================== #

@pytest.mark.parametrize("port", ["reflection", "transmission"])
def test_s5_5_rcwa_method_equals_free_function(port):
    """RCWAResult.to_multiorder_field now delegates to jones_field_from_orders
    on its public per_order_amplitudes(port) dict -- byte-for-byte identical,
    which also proves the augmented dict carries everything the bridge needs."""
    res = _rcwa_grating(np.deg2rad(12.0)).solve()
    amps = res.per_order_amplitudes(port)
    nx = 60
    m = res.to_multiorder_field(nx, nx, _PERIOD / nx, incident=(1.0, 0.0),
                                port=port, normalize="power")
    f = jones_field_from_orders(amps, nx, nx, _PERIOD / nx,
                                incident=(1.0, 0.0), normalize="power")
    assert np.array_equal(m.Ex, f.Ex) and np.array_equal(m.Ey, f.Ey)
    # independent Parseval oracle on the delegated result
    _o, R, T = res.efficiencies()
    e_tot = _eff_total((R if port == "reflection" else T)[0], amps["kz"])
    assert abs(_field_power(f) - e_tot) < 1e-6


def test_s5_5_pmm_and_rcwa_bridge_agree():
    """Two independent solvers, one shared bridge: the reconstructed diffracted
    power agrees to the engines' amplitude-parity tolerance."""
    th = np.deg2rad(15.0)
    sp = _pmm_grating(th)
    sp.solve()
    rr = _rcwa_grating(th).solve()
    nx = 64
    for port in ("reflection", "transmission"):
        fp = jones_field_from_orders(sp.per_order_amplitudes(port), nx, nx,
                                     _PERIOD / nx, normalize="power")
        fr = jones_field_from_orders(rr.per_order_amplitudes(port), nx, nx,
                                     _PERIOD / nx, normalize="power")
        assert abs(_field_power(fp) - _field_power(fr)) < 5e-3


# ======================================================================== #
# Guards + the power/field distinction.                                    #
# ======================================================================== #

def test_s5_5_bridge_guards_and_field_mode():
    res = _rcwa_grating(np.deg2rad(12.0)).solve()
    amps = res.per_order_amplitudes("transmission")
    # the augmentation is load-bearing: without kz_inc the power weight is
    # ill-defined -> an actionable error, not a silent wrong field.
    bad = dict(amps)
    bad.pop("kz_inc")
    with pytest.raises(ValueError, match="missing 'kz_inc'"):
        jones_field_from_orders(bad, 32, 32, _PERIOD / 32)
    with pytest.raises(ValueError, match="normalize must be"):
        jones_field_from_orders(amps, 32, 32, _PERIOD / 32, normalize="bogus")
    with pytest.raises(ValueError, match="filter must be"):
        jones_field_from_orders(amps, 32, 32, _PERIOD / 32, filter="bogus")
    # raw 'field' mode drops the Poynting flux weight + the longitudinal Ez, so
    # at oblique incidence its power is NOT the diffraction efficiency.
    _o, R, T = res.efficiencies()
    e_tot = _eff_total(T[0], amps["kz"])
    nx = 60
    p_pow = _field_power(jones_field_from_orders(
        amps, nx, nx, _PERIOD / nx, normalize="power"))
    p_fld = _field_power(jones_field_from_orders(
        amps, nx, nx, _PERIOD / nx, normalize="field"))
    assert abs(p_pow - e_tot) < 1e-6
    assert abs(p_fld - e_tot) > 1e-4


def test_s5_5_explicit_evanescent_order_skipped():
    """An explicitly requested evanescent order warns and is skipped rather
    than depositing a bogus fast-oscillating carrier."""
    res = _rcwa_grating(np.deg2rad(12.0)).solve()
    amps = res.per_order_amplitudes("reflection")   # n_sup=1 -> only order 0
    assert np.real(amps["kz"])[np.asarray(amps["orders"]) == 5][0] <= 1e-12
    with pytest.warns(UserWarning, match="evanescent"):
        jf = jones_field_from_orders(amps, 32, 32, _PERIOD / 32, orders=[5])
    assert np.max(np.abs(jf.Ex)) == 0.0 and np.max(np.abs(jf.Ey)) == 0.0
