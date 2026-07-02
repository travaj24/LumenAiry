"""Audit v5.17.0 P1-01: BOR flux-normalization threshold must be RELATIVE.

zcascade._layer_modes_staggered / bor_solve._flux_normalize used an ABSOLUTE
flux threshold (1e-10) to split unit-|z-flux| normalization (propagating) from
the field-norm fallback (evanescent).  The z-flux is dimensional (scales as
field^2 * length^2), so the same physics expressed in meters instead of microns
pushed every propagating mode's flux (~Rbig^2/(2N) ~ 4e-13) under the
threshold: all modes silently took the field-norm fallback, |S|^2 stopped
being a power fraction, and BORStack.solve returned wrong R/T (energy
0.44..1.00 instead of 1).  Fixed by comparing the flux against the mode's own
r*dr field norm (same measure, same scaling -> unit-invariant).

Pre-fix, test_si_units_energy_conserved, test_micron_vs_meter_rt_identical and
test_flux_normalize_si_scale all FAIL (verified via git-stash A/B);
test_evanescent_fallback_preserved guards against overcorrection.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from lumenairy.elements.bor import BORStack
from lumenairy.elements.bor.bor_solve import _flux_normalize
from lumenairy.elements.bor.coupled_radial_eigensolver import _fd_grid_staggered
from lumenairy.elements.bor.zcascade import layer_modes


def _solve(scale):
    """Same lossless ring-grating stack at unit length = 1/scale meters
    (scale=1.0 -> micron-style O(1) units; scale=1e-6 -> SI meters)."""
    s = BORStack(Rbig=8.0 * scale, m=1, N=80, n_superstrate=1.0,
                 n_substrate=1.5)
    s.add_layer(0.5 * scale, rings=(2.0 * scale, 0.5, 1.5, 1.0))
    s.set_source(k0=2.0 / scale)
    return s.solve()


def test_si_units_energy_conserved():
    """SI (meter-scale) lossless stack: R+T must be 1 for every incident mode.
    Pre-fix: energy = [0.67, 0.69, ..., 0.44] (field-norm fallback fired)."""
    res = _solve(1e-6)
    assert len(res["inc"]) == 9
    assert np.max(np.abs(res["energy"] - 1.0)) < 1e-6


def test_micron_vs_meter_rt_identical():
    """Maxwell is scale-invariant: identical physics in micron-style units and
    SI meters must give the same R/T.  Pre-fix: max|T_um - T_SI| = 0.55."""
    res_um = _solve(1.0)
    res_si = _solve(1e-6)
    assert len(res_um["inc"]) == len(res_si["inc"])
    assert np.max(np.abs(res_um["R"] - res_si["R"])) < 1e-10
    assert np.max(np.abs(res_um["T"] - res_si["T"])) < 1e-10
    assert np.max(np.abs(res_um["energy"] - 1.0)) < 1e-10
    assert np.max(np.abs(res_si["energy"] - 1.0)) < 1e-10


def _uniform(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _fluxes_staggered(L, Rbig):
    """True staggered z-flux: Er*hphi pairs on FACES, Ephi*hr on NODES."""
    N = L["N"]
    r_n, r_f, h = _fd_grid_staggered(Rbig, N)[:3]
    return np.array([np.real(
        np.sum(L["W"][:N, j] * np.conj(L["V"][N:, j]) * r_f * h)
        - np.sum(L["W"][N:, j] * np.conj(L["V"][:N, j]) * r_n * h))
        for j in range(2 * N)])


def _count_unit_flux(scale):
    L = _flux_normalize(layer_modes(1, 6.0 * scale, 60, _uniform(1.0),
                                    2.0 / scale, wall="pec"))
    return np.count_nonzero(np.abs(np.abs(L["flux"]) - 1.0) < 1e-6)


def test_flux_normalize_si_scale():
    """bor_solve._flux_normalize (the duplicated site): the SET of modes that
    get unit-|z-flux| normalization must be scale-invariant.  Pre-fix at SI
    scale ZERO modes are unit-flux (all post-norm fluxes ~ 1e-13: every mode
    took the field-norm fallback)."""
    n_um = _count_unit_flux(1.0)
    n_si = _count_unit_flux(1e-6)
    assert n_um > 0
    assert n_si == n_um


def test_evanescent_fallback_preserved():
    """Genuinely evanescent modes (true flux ~ 0 relative to field norm) must
    STILL take the field-norm fallback -- at BOTH scales.  Propagating modes
    must be unit-flux.  Guards against overcorrecting the threshold."""
    for scale in (1.0, 1e-6):
        L = layer_modes(1, 8.0 * scale, 80, _uniform(1.0), 2.0 / scale,
                        staggered=True)
        q, fx = L["q"], _fluxes_staggered(L, 8.0 * scale)
        prop = np.abs(q.imag) < 1e-9 * np.maximum(np.abs(q.real), 1e-300)
        assert np.count_nonzero(prop) == 9
        # propagating: exactly unit flux after normalization
        assert np.max(np.abs(np.abs(fx[prop]) - 1.0)) < 1e-9
        # evanescent: field-norm fallback -> |flux| stays ~0, never ~1
        assert np.max(np.abs(fx[~prop])) < 1e-6
        # fallback == unit plain field norm (the evanescent convention)
        ev = np.where(~prop)[0][0]
        fn = np.sum(np.abs(L["W"][:, ev]) ** 2)
        assert abs(fn - 1.0) < 1e-9


def test_nm_scale_units_classified_and_conserved():
    """Audit P2-06 (found by the wave-1 adversarial pass): the propagating-mode
    classifiers used ABSOLUTE q thresholds (|q.imag| < 1e-4, q.real > 0.1), so
    nm-style small-k0 unit systems (k0 = 2e-3) silently returned inc=[], R=[],
    T=[].  Classifiers now work in the dimensionless q/k0."""
    res_um = _solve(1.0)
    for scale in (1e3, 1e4):                 # nm-style and coarser
        res = _solve(scale)
        assert len(res["inc"]) == len(res_um["inc"])   # pre-fix: 0
        assert np.max(np.abs(res["R"] - res_um["R"])) < 1e-9
        assert np.max(np.abs(res["T"] - res_um["T"])) < 1e-9
        assert np.max(np.abs(res["energy"] - 1.0)) < 1e-9
