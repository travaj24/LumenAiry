"""Niche R5 (roadmap A7 + A4).

A7 -- GBD VECTOR PRESCRIPTION-CHAIN CARRIER PORT.
    ``propagate_gbd_vector_through_prescription`` gets ``direction_sampling=
    'auto'`` (default), porting the N4 Husimi carrier decomposition -- the SHARED
    ``_input_angular_spread`` detector + auto logic -- to the vector (Jones)
    chain.  A collimated Jones input measures ~0 spread and takes the
    byte-identical axial path; a diverging / converging / tilted input launches
    each beamlet along the field's local wavevector and conserves power (to the
    Fresnel-transmission limit) instead of collapsing to the collimated plane and
    shedding frame energy.

    The vector chain scatters the Fresnel-mixed samples onto full grids and runs
    two scalar prescription chains; the historical zero-scatter grid is near-empty
    at ``sample_step > 1`` and its gradient is dominated by the fill, so the
    Husimi launch collapsed to axial (power stuck at ~0.75 even with
    ``direction_sampling=True``).  A7 builds a DENSE mixed field (smooth
    Fresnel-Jones bilinearly upsampled x the dense input) for the Husimi launch so
    the carrier-normal decomposition reads the true wavefront.

    Oracles: (1) chain-vs-sequential equivalence -- the vector chain matches, per
    component, both back-to-back scalar prescription chains on the SAME mixed
    fields (by construction) AND ``apply_real_lens_gbd`` on those fields (the
    independent single-entry H7 reference, mirroring N4); (2) diverging-input
    power > 0.99 of the Fresnel-transmission limit, fail-before (axial < 0.9) /
    pass-after; (3) collimated byte-identical (auto == axial).

A4 -- HIGHER-CATASTROPHE DETECT + ROUTE.
    The ``caustic='uniform'`` dispatcher classifies the catastrophe from the
    meridional map's interior-turning-point count (1 fold, 2 cusp, 3 swallowtail,
    4 butterfly, >=5 non-classifiable).  A swallowtail / butterfly /
    non-classifiable higher catastrophe (``n_turn >= 3``) has no canonical
    completion here; it is DETECTED and routed CLEANLY to the finite multibranch
    fallback with a one-time NAMED warning (never inf/nan).  Detector test on
    synthetic higher-catastrophe meridional maps + a monkeypatched live route.

Oracle independence: the A7 power metric is Gaussian-beam power conservation +
the Fresnel transmittance from the per-beamlet Jones matrix (both independent of
the GBD reconstruction); the A4 detector is a pure turning-point count on a
synthetic map.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
import lumenairy.elements._lens_traced_uniform as umod
from lumenairy.elements._lens_traced_uniform import (
    _classify_catastrophe,
    _count_interior_turning_points,
    apply_real_lens_traced_uniform,
)
from lumenairy.propagators.gbd import (
    _fresnel_jones_matrix_per_beamlet,
    _vector_prescription_mixed_fields,
    propagate_gbd_through_prescription,
    propagate_gbd_vector_through_prescription,
)

_WL = 1.31e-6
# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_R5_GLASS': lambda wl: 1.5168,
                  '_R5_162': lambda wl: 1.62}

# Compact config: reconstruct at the EXIT vertex (z_image=0.0) where the beamlets
# are still compact, so the windowed reconstruct is fast and power is well-posed.
_N, _DX, _W_L, _STEP = 256, 8e-6, 1.0e-3, 4


# ---------------------------------------------------------------------------
# prescriptions + fields
# ---------------------------------------------------------------------------
def _singlet():
    return {'wavelength': _WL, 'aperture_diameter': 20e-3,
            'surfaces': [
                {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
                 'glass_after': '_R5_GLASS', 'semi_diameter': 10e-3},
                {'radius': -51.68e-3, 'thickness': 0.0,
                 'glass_before': '_R5_GLASS', 'glass_after': 'air',
                 'semi_diameter': 10e-3}],
            'thicknesses': [5e-3], 'stop_index': 0}


def _doublet():
    return {'wavelength': _WL, 'aperture_diameter': 40e-3,
            'surfaces': [
                {'radius': 61.5e-3, 'thickness': 6e-3, 'glass_before': 'air',
                 'glass_after': '_R5_GLASS', 'semi_diameter': 20e-3},
                {'radius': -44.5e-3, 'thickness': 2.5e-3,
                 'glass_before': '_R5_GLASS', 'glass_after': '_R5_162',
                 'semi_diameter': 20e-3},
                {'radius': -129e-3, 'thickness': 0.0,
                 'glass_before': '_R5_162', 'glass_after': 'air',
                 'semi_diameter': 20e-3}],
            'thicknesses': [6e-3, 2.5e-3], 'stop_index': 0}


def _diverging(N, dx, w_L, R_in):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_sq = X ** 2 + Y ** 2
    k = 2 * np.pi / _WL
    return (np.exp(-r_sq / w_L ** 2)
            * np.exp(1j * k * r_sq / (2.0 * R_in))).astype(np.complex128)


def _collimated(N, dx, w_L):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w_L ** 2).astype(np.complex128)


def _vchain(E_vec, presc, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(propagate_gbd_vector_through_prescription(
            E_vec, _DX, presc, wavelength=_WL, per_surface=True, z_image=0.0,
            sample_step=_STEP, waist_factor=float(_STEP), **kw))


def _power(V):
    return float(np.sum(np.abs(V) ** 2))


def _fresnel_T(E_vec, presc):
    """Beamlet Fresnel transmittance T = sum|P E|^2 / sum|E|^2 over the subsample
    lattice -- the physical power ceiling for the vector chain (each transmitting
    surface sheds ~4% by Fresnel; a diverging singlet passes ~0.92).  Independent
    of the GBD reconstruction (just the per-beamlet Jones matrix)."""
    Ny, Nx = E_vec.shape[-2], E_vec.shape[-1]
    iy = np.arange(0, Ny, _STEP)
    ix = np.arange(0, Nx, _STEP)
    Iy, Ix = np.meshgrid(iy, ix, indexing='ij')
    xb = (Ix.ravel() - Nx / 2) * _DX
    yb = (Iy.ravel() - Ny / 2) * _DX
    zc = np.zeros_like(xb)
    P, _ = _fresnel_jones_matrix_per_beamlet(xb, yb, zc, zc, presc, _WL)
    ExS = E_vec[0][Iy, Ix].ravel()
    EyS = E_vec[1][Iy, Ix].ravel()
    ExM = P[:, 0, 0] * ExS + P[:, 0, 1] * EyS
    EyM = P[:, 1, 0] * ExS + P[:, 1, 1] * EyS
    num = float(np.sum(np.abs(ExM) ** 2 + np.abs(EyM) ** 2))
    den = float(np.sum(np.abs(ExS) ** 2 + np.abs(EyS) ** 2))
    return num / den


# ===========================================================================
# A7.1  Collimated byte-identical: auto resolves to the axial path.
# ===========================================================================
def test_a7_collimated_auto_is_axial_byte_identical():
    """A collimated Jones input measures ~0 angular spread, so ``'auto'``
    resolves to the axial launch -- identical to ``direction_sampling=False``
    (the historical default).  Asserted with a tolerance (<= 1e-10 * scale), not
    array_equal, per the cache-cold/warm ULP discipline."""
    Ex = _collimated(_N, _DX, _W_L)
    E_vec = np.stack([Ex, 0.3 * Ex], axis=0)      # uniformly polarized, flat
    E_auto = _vchain(E_vec, _singlet(), direction_sampling='auto')
    E_axial = _vchain(E_vec, _singlet(), direction_sampling=False)
    peak = float(np.abs(E_axial).max())
    md = float(np.max(np.abs(E_auto - E_axial)))
    assert md <= 1e-10 * peak, (
        f"collimated auto default diverged from the axial path (max|diff|={md:.2e},"
        f" peak={peak:.2e})")
    # and the shared detector indeed resolved axial
    _, _, ds = _vector_prescription_mixed_fields(
        E_vec, _DX, _singlet(), wavelength=_WL, sample_step=_STEP,
        direction_sampling='auto')
    assert ds is False


# ===========================================================================
# A7.2  Fail-before: the axial launch sheds the diverging beam's power.
# ===========================================================================
def test_a7_axial_default_sheds_diverging_power():
    """With ``direction_sampling=False`` the diverging Jones beam's curvature
    rides the beamlet amplitude only; the axial base rays focus at the collimated
    plane and the frame sheds the beam's power well below the Fresnel limit."""
    Ex = _diverging(_N, _DX, _W_L, 150e-3)
    E_vec = np.stack([Ex, np.zeros_like(Ex)], axis=0)
    T = _fresnel_T(E_vec, _singlet())
    pc = _power(_vchain(E_vec, _singlet(), direction_sampling=False)) / _power(E_vec)
    assert pc / T < 0.9, (
        f"axial chain should shed the diverging beam's power (got {pc/T:.4f} of "
        f"the Fresnel limit T={T:.4f}); if it no longer collapses the fail-before "
        f"is moot")


# ===========================================================================
# A7.3  Pass-after: the 'auto' default conserves power to the Fresnel limit.
# ===========================================================================
@pytest.mark.parametrize('R_in', [300e-3, 150e-3])
def test_a7_auto_conserves_diverging_power(R_in):
    """The ported Husimi launch rides the diverging Jones input's curvature and
    conserves power to > 0.99 of the Fresnel-transmission limit (the vector chain
    ceiling; the scalar N4 chain, with no Fresnel loss, reaches > 0.99 of unity).
    Finite everywhere."""
    Ex = _diverging(_N, _DX, _W_L, R_in)
    E_vec = np.stack([Ex, np.zeros_like(Ex)], axis=0)
    T = _fresnel_T(E_vec, _singlet())
    E = _vchain(E_vec, _singlet(), direction_sampling='auto')
    assert np.all(np.isfinite(E)), "non-finite entries in the vector chain output"
    pc = _power(E) / _power(E_vec)
    assert pc / T > 0.99, (
        f"R_in={R_in*1e3:.0f}mm: vector-chain auto power {pc:.4f} is only "
        f"{pc/T:.4f} of the Fresnel limit T={T:.4f} (need > 0.99) -- the Husimi "
        f"launch must ride the diverging input's curvature")


def test_a7_doublet_diverging_conserves_power():
    """Multi-element generality: an M1-class doublet + diverging Jones input
    conserves power > 0.99 of the Fresnel limit under the new default."""
    Ex = _diverging(_N, _DX, _W_L, 150e-3)
    E_vec = np.stack([Ex, np.zeros_like(Ex)], axis=0)
    T = _fresnel_T(E_vec, _doublet())
    E = _vchain(E_vec, _doublet(), direction_sampling='auto')
    assert np.all(np.isfinite(E))
    pc = _power(E) / _power(E_vec)
    assert pc / T > 0.99, (
        f"doublet vector-chain diverging power {pc/T:.4f} of Fresnel limit "
        f"T={T:.4f} <= 0.99")


# ===========================================================================
# A7.4  Chain-vs-sequential equivalence (by construction): the vector chain
#       equals two back-to-back scalar prescription chains on the SAME mixed
#       fields.  (Two live calls -> tolerance, not array_equal.)
# ===========================================================================
@pytest.mark.parametrize('presc_fn', [_singlet, _doublet])
def test_a7_chain_matches_sequential_scalar(presc_fn):
    presc = presc_fn()
    Ex = _diverging(_N, _DX, _W_L, 150e-3)
    E_vec = np.stack([Ex, 0.25 * Ex], axis=0)     # nontrivial two-component
    V = _vchain(E_vec, presc, direction_sampling='auto')
    ExF, EyF, ds = _vector_prescription_mixed_fields(
        E_vec, _DX, presc, wavelength=_WL, sample_step=_STEP,
        direction_sampling='auto')
    assert ds is True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rx = np.asarray(propagate_gbd_through_prescription(
            ExF, _DX, presc, wavelength=_WL, per_surface=True, z_image=0.0,
            sample_step=_STEP, waist_factor=float(_STEP), direction_sampling=ds))
        ry = np.asarray(propagate_gbd_through_prescription(
            EyF, _DX, presc, wavelength=_WL, per_surface=True, z_image=0.0,
            sample_step=_STEP, waist_factor=float(_STEP), direction_sampling=ds))
    peak = max(float(np.abs(V).max()), 1e-30)
    md = max(float(np.max(np.abs(V[0] - rx))), float(np.max(np.abs(V[1] - ry))))
    assert md <= 1e-10 * peak, (
        f"vector chain != back-to-back scalar chains on the mixed fields "
        f"(max|diff|={md:.2e}, peak={peak:.2e})")


# ===========================================================================
# A7.5  Independent equivalence: the vector chain matches apply_real_lens_gbd
#       (the single-entry H7 reference) per component on the SAME mixed fields
#       -- two DIFFERENT code paths agreeing, mirroring N4's scalar equivalence.
# ===========================================================================
def test_a7_chain_matches_apply_real_lens_gbd():
    presc = _singlet()
    Ex = _diverging(_N, _DX, _W_L, 150e-3)
    E_vec = np.stack([Ex, np.zeros_like(Ex)], axis=0)
    V = _vchain(E_vec, presc, direction_sampling='auto')
    ExF, EyF, ds = _vector_prescription_mixed_fields(
        E_vec, _DX, presc, wavelength=_WL, sample_step=_STEP,
        direction_sampling='auto')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = np.asarray(la.apply_real_lens_gbd(
            ExF, prescription=presc, wavelength=_WL, dx=_DX, sample_step=_STEP,
            waist_factor=float(_STEP), window=5.0, clip_aperture=False,
            output_subsample=1, direction_sampling=ds))
        ay = np.asarray(la.apply_real_lens_gbd(
            EyF, prescription=presc, wavelength=_WL, dx=_DX, sample_step=_STEP,
            waist_factor=float(_STEP), window=5.0, clip_aperture=False,
            output_subsample=1, direction_sampling=ds))
    m = np.abs(ax) > 1e-3 * np.abs(ax).max()
    rel = (float(np.linalg.norm((V[0] - ax)[m]))
           / (float(np.linalg.norm(ax[m])) + 1e-30))
    assert rel < 1e-6, (
        f"vector-chain x-component vs apply_real_lens_gbd rel L2 = {rel:.2e} "
        f"(expected numerical-precision agreement with matched params)")
    # the (near-zero) cross-pol y-component also agrees where it has support
    my = np.abs(ay) > 1e-3 * np.abs(ax).max()
    if my.any():
        rely = (float(np.linalg.norm((V[1] - ay)[my]))
                / (float(np.linalg.norm(ay[my])) + 1e-30))
        assert rely < 1e-6, rely


# ===========================================================================
# A7.6  Shared detector: a MIXED-pol input (flat x, diverging y) fires Husimi
#       for the WHOLE field (max spread over the two components), so the
#       diverging component's power is conserved.
# ===========================================================================
def test_a7_shared_detector_fires_on_mixed_pol():
    Ex_flat = _collimated(_N, _DX, _W_L)
    Ey_div = _diverging(_N, _DX, _W_L, 150e-3)
    E_vec = np.stack([Ex_flat, Ey_div], axis=0)
    # the shared (max-over-components) detector must fire even though x is flat
    _, _, ds = _vector_prescription_mixed_fields(
        E_vec, _DX, _singlet(), wavelength=_WL, sample_step=_STEP,
        direction_sampling='auto')
    assert ds is True, "shared detector must fire Husimi when EITHER component curves"
    T = _fresnel_T(E_vec, _singlet())
    E_auto = _vchain(E_vec, _singlet(), direction_sampling='auto')
    E_axial = _vchain(E_vec, _singlet(), direction_sampling=False)
    assert np.all(np.isfinite(E_auto))
    p_auto = _power(E_auto) / _power(E_vec)
    p_axial = _power(E_axial) / _power(E_vec)
    assert p_auto / T > 0.99, (p_auto, T)
    assert p_auto > p_axial + 0.05, (p_auto, p_axial)


# ===========================================================================
# A7.7  Guard: an invalid direction_sampling policy raises (mirrors the scalar
#       chain contract).
# ===========================================================================
def test_a7_invalid_direction_sampling_raises():
    Ex = _diverging(64, _DX, _W_L, 150e-3)
    E_vec = np.stack([Ex, np.zeros_like(Ex)], axis=0)
    with pytest.raises(ValueError, match="direction_sampling"):
        propagate_gbd_vector_through_prescription(
            E_vec, _DX, _singlet(), wavelength=_WL, per_surface=True,
            sample_step=8, waist_factor=8.0, direction_sampling='bogus')


# ===========================================================================
# A4.1  Detector on synthetic higher-catastrophe meridional maps.
# ===========================================================================
def test_a4_turning_point_counter_synthetic():
    """A synthetic meridional map ``x_out(h) = sin(k pi h)`` has exactly ``k``
    interior turning points (fold caustics); the detector counts them robustly
    (no zero-slope double-count) and the classifier names the catastrophe."""
    h = np.linspace(0.0, 1.0, 1200)
    expected = {1: 'fold (A2)', 2: 'cusp (A3)', 3: 'swallowtail (A4)',
                4: 'butterfly (A5)'}
    for k in range(1, 5):
        n = _count_interior_turning_points(np.sin(k * np.pi * h))
        assert n == k, (k, n)
        assert _classify_catastrophe(n) == expected[k]
    # >= 5 folds -> a non-classifiable higher catastrophe
    n5 = _count_interior_turning_points(np.sin(5 * np.pi * h))
    assert n5 == 5
    assert 'non-classifiable' in _classify_catastrophe(n5)
    # a degree-4 polynomial with three humps is a swallowtail
    hh = np.linspace(0.0, 4.0, 1500)
    poly = np.poly1d(np.polyfit([0, 1, 2, 3, 4], [0, 1, -1, 1, 0], 4))
    assert _count_interior_turning_points(poly(hh)) == 3
    # degenerate short maps never crash
    assert _count_interior_turning_points([1.0]) == 0
    assert _count_interior_turning_points([1.0, 2.0]) == 0


def test_a4_classify_names_hierarchy():
    assert _classify_catastrophe(1) == 'fold (A2)'
    assert _classify_catastrophe(2) == 'cusp (A3)'
    assert _classify_catastrophe(3) == 'swallowtail (A4)'
    assert _classify_catastrophe(4) == 'butterfly (A5)'
    assert 'non-classifiable' in _classify_catastrophe(6)


# ===========================================================================
# A4.2  Live route: a higher catastrophe (n_turn >= 3) routes CLEANLY to the
#       finite multibranch fallback with a one-time NAMED warning (no inf/nan).
# ===========================================================================
@pytest.mark.parametrize('n_turn,name', [
    (3, 'swallowtail'), (4, 'butterfly'), (5, 'non-classifiable')])
def test_a4_higher_catastrophe_routes_finite_named(monkeypatch, n_turn, name):
    """When the meridional trace reports ``n_turn >= 3`` the uniform dispatcher
    detects the class, routes to the plain (finite) multibranch field, and warns
    NAMING the class -- never inf/nan, never the cusp (Pearcey) path."""
    N, dx = 128, 2e-6
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (0.1e-3) ** 2).astype(np.complex128)
    base = (np.cos(np.hypot(X, Y) / dx) + 0.5).astype(np.complex128)  # finite

    def fake_mb(E_in, **kw):
        if kw.get('return_diagnostics'):
            return base, {'input_carrier': (0.0, 0.0)}
        return base
    monkeypatch.setattr(umod, 'apply_real_lens_traced_multibranch', fake_mb)
    monkeypatch.setattr(umod, '_is_rotationally_symmetric', lambda *a, **k: True)
    monkeypatch.setattr(
        umod, '_trace_meridional_fold',
        lambda *a, **k: {'ok': False, 'reason': 'cusp_or_multiple',
                         'n_turn': n_turn})
    # the cusp (Pearcey) trace must NOT be consulted for n_turn >= 3
    def _boom(*a, **k):
        raise AssertionError('cusp trace should not run for a higher catastrophe')
    monkeypatch.setattr(umod, '_trace_meridional_cusp', _boom)

    presc = {'wavelength': _WL, 'surfaces': [], 'thicknesses': []}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        E, diag = apply_real_lens_traced_uniform(
            E0, prescription=presc, wavelength=_WL, dx=dx,
            output_plane_distance=1e-3, return_diagnostics=True)
    assert np.all(np.isfinite(E))            # never inf/nan
    assert np.array_equal(E, base)           # the finite multibranch fallback
    assert diag['fell_back'] is True
    assert name in diag['reason'], diag['reason']
    msgs = [str(m.message) for m in w]
    assert any(name in m for m in msgs), msgs   # the warning NAMES the class


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
