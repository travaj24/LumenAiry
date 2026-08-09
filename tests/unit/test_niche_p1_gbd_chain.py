"""Accuracy-niches P1 / N4 (2026-07-19): GBD prescription-chain carrier
extension.

The hammer H7 fix (carrier-normal / Husimi beamlet launch for a diverging /
converging / tilted input) landed at the ``apply_real_lens_gbd`` entry only.
The prescription-chain entry :func:`propagate_gbd_through_prescription` still
defaulted to the position-only (axial) decomposition, so a diverging input's
wavefront curvature rode the beamlet AMPLITUDE only: the axial base rays focused
at the collimated plane and the frame shed the diverging beam's energy (power
collapsed to ~0.2-0.8 on the dual-oracle singlet).

N4 ports the same ``direction_sampling='auto'`` policy into the chain entry
(default), resolved through the SHARED ``_input_angular_spread`` /
``_GBD_AUTO_HUSIMI_THRESH`` now defined in the beamlet layer.  A diverging input
then conserves power; a flat-wavefront (collimated) input measures ~0 spread and
takes the byte-identical axial path.

Oracle: ABCD Gaussian q-trace through the dual-oracle singlet and an M1-class
doublet -- independent of the GBD implementation; here the power-conservation
metric at the EXIT vertex (independent of any focus assumption).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.gbd import propagate_gbd_through_prescription

_WL = 1.31e-6
# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_N4_FIX_GLASS': lambda wl: 1.5168,
                  '_N4_FIX_162': lambda wl: 1.62}

# Compact config: reconstruct at the EXIT vertex (z_image=0.0) where the beamlets
# are still compact, so the windowed reconstruct is fast and power is well-posed.
_N, _DX, _W_L, _STEP = 256, 8e-6, 1.0e-3, 4


def _singlet():
    return {'wavelength': _WL, 'aperture_diameter': 20e-3,
            'surfaces': [
                {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
                 'glass_after': '_N4_FIX_GLASS', 'semi_diameter': 10e-3},
                {'radius': -51.68e-3, 'thickness': 0.0,
                 'glass_before': '_N4_FIX_GLASS', 'glass_after': 'air',
                 'semi_diameter': 10e-3}],
            'thicknesses': [5e-3], 'stop_index': 0}


def _doublet():
    return {'wavelength': _WL, 'aperture_diameter': 40e-3,
            'surfaces': [
                {'radius': 61.5e-3, 'thickness': 6e-3, 'glass_before': 'air',
                 'glass_after': '_N4_FIX_GLASS', 'semi_diameter': 20e-3},
                {'radius': -44.5e-3, 'thickness': 2.5e-3,
                 'glass_before': '_N4_FIX_GLASS', 'glass_after': '_N4_FIX_162',
                 'semi_diameter': 20e-3},
                {'radius': -129e-3, 'thickness': 0.0,
                 'glass_before': '_N4_FIX_162', 'glass_after': 'air',
                 'semi_diameter': 20e-3}],
            'thicknesses': [6e-3, 2.5e-3], 'stop_index': 0}


def _diverging_input(N, dx, w_L, R_in):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    k = 2 * np.pi / _WL
    return (np.exp(-r_sq / w_L ** 2)
            * np.exp(1j * k * r_sq / (2.0 * R_in))).astype(np.complex128)


def _chain(E0, presc, ds='auto'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(propagate_gbd_through_prescription(
            E0, _DX, presc, wavelength=_WL, per_surface=True, z_image=0.0,
            sample_step=_STEP, waist_factor=float(_STEP), direction_sampling=ds))


def _power(E):
    return float(np.sum(np.abs(E) ** 2))


# ---------------------------------------------------------------------------
# Fail-before: the pre-N4 axial default sheds the diverging beam's power.
# ---------------------------------------------------------------------------
def test_n4_axial_default_sheds_diverging_power():
    E0 = _diverging_input(_N, _DX, _W_L, 150e-3)
    pc = _power(_chain(E0, _singlet(), ds=False)) / _power(E0)
    assert pc < 0.9, (
        f"axial (pre-N4) chain entry should shed the diverging beam's power "
        f"(got {pc:.4f}); if it no longer collapses the fail-before is moot")


# ---------------------------------------------------------------------------
# Pass-after: the new 'auto' default conserves power and stays finite.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('R_in', [300e-3, 150e-3])
def test_n4_auto_default_conserves_diverging_power(R_in):
    E0 = _diverging_input(_N, _DX, _W_L, R_in)
    E = _chain(E0, _singlet(), ds='auto')
    assert np.all(np.isfinite(E)), "non-finite entries in the chain output"
    pc = _power(E) / _power(E0)
    assert pc > 0.99, (
        f"R_in={R_in*1e3:.0f}mm: chain-entry auto power {pc:.4f} <= 0.99 -- the "
        f"Husimi launch must ride the diverging input's curvature")


# ---------------------------------------------------------------------------
# Equivalence: the chain entry matches apply_real_lens_gbd (the H7-fixed single
# entry) to numerical precision on a two-group system (doublet), matched params
# -- both single-decompose the SAME auto/Husimi frame and share the per-surface
# evolve + windowed reconstruct, so the fields agree bit-for-bit.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('presc_fn', [_singlet, _doublet])
def test_n4_chain_matches_apply_real_lens_gbd(presc_fn):
    presc = presc_fn()
    E0 = _diverging_input(_N, _DX, _W_L, 150e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_arl = np.asarray(la.apply_real_lens_gbd(
            E0, prescription=presc, wavelength=_WL, dx=_DX, sample_step=_STEP,
            waist_factor=float(_STEP), window=5.0, clip_aperture=False,
            output_subsample=1))
    E_chain = _chain(E0, presc, ds='auto')
    m = np.abs(E_arl) > 1e-3 * np.abs(E_arl).max()
    rel = (float(np.linalg.norm((E_chain - E_arl)[m]))
           / (float(np.linalg.norm(E_arl[m])) + 1e-30))
    assert rel < 1e-9, (
        f"chain entry vs apply_real_lens_gbd rel L2 = {rel:.2e} (expected "
        f"numerical-precision agreement with matched params)")


# ---------------------------------------------------------------------------
# Byte-identical: a collimated input measures ~0 angular spread, so 'auto'
# resolves to the axial path -- identical to direction_sampling=False.
# ---------------------------------------------------------------------------
def test_n4_collimated_auto_is_axial_byte_identical():
    x = (np.arange(_N) - _N / 2) * _DX
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / _W_L ** 2).astype(np.complex128)
    E_auto = _chain(E0, _singlet(), ds='auto')
    E_axial = _chain(E0, _singlet(), ds=False)
    assert np.array_equal(E_auto, E_axial), (
        "collimated auto default diverged from the axial path "
        f"(max|diff|={np.max(np.abs(E_auto - E_axial)):.2e})")


# ---------------------------------------------------------------------------
# M1-class doublet, diverging input: the chain entry (new default) conserves
# power > 0.99 -- multi-element generality of the ported carrier decomposition.
# ---------------------------------------------------------------------------
def test_n4_doublet_diverging_conserves_power():
    E0 = _diverging_input(_N, _DX, _W_L, 150e-3)
    E = _chain(E0, _doublet(), ds='auto')
    assert np.all(np.isfinite(E))
    pc = _power(E) / _power(E0)
    assert pc > 0.99, f"doublet chain-entry diverging power {pc:.4f} <= 0.99"


# ---------------------------------------------------------------------------
# Guard: an invalid direction_sampling policy raises (mirrors
# apply_real_lens_gbd's contract).
# ---------------------------------------------------------------------------
def test_n4_invalid_direction_sampling_raises():
    E0 = _diverging_input(64, _DX, _W_L, 150e-3)
    with pytest.raises(ValueError, match="direction_sampling"):
        propagate_gbd_through_prescription(
            E0, _DX, _singlet(), wavelength=_WL, per_surface=True, z_image=0.0,
            sample_step=8, waist_factor=8.0, direction_sampling='bogus')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
