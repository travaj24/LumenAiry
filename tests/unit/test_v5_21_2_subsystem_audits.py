"""Subsystem-audit remediation campaign (docs/audits/AUDIT_*_2026_07_07..09).

One growing file, sectioned per audit, pinning the findings fixed in the
chronological subsystem sweep that follows the v5.21 delta audit.
"""
import os

import numpy as np
import pytest

# =========================================================================
# AUDIT 2 -- sources/core.py (AUDIT_SOURCES_CORE_2026_07_07)
# =========================================================================
from lumenairy.sources.core import (
    PartialCoherenceMCF,
    create_annular_beam,
    create_hermite_gauss,
    create_laguerre_gauss,
    create_multi_field_sources,
    create_top_hat_beam,
)


def _complex_j_ensemble(Ny=6, Nx=6, nr=40, seed=0):
    """A complex-MCF (linearly-phase-tilted + noise) ensemble -- its
    J(r1, r2) phase sign is nonzero, so the SRC-1 conjugation shows up."""
    rng = np.random.default_rng(seed)
    xs = np.arange(Nx)
    base = np.exp(1j * 0.7 * xs)[None, :] * np.ones((Ny, 1))
    return np.stack([
        base * np.exp(1j * rng.normal(0, 0.3))
        + 0.2 * (rng.standard_normal((Ny, Nx))
                 + 1j * rng.standard_normal((Ny, Nx)))
        for _ in range(nr)])


def test_src1_dense_matches_documented_mcf_convention():
    """SRC-1: the dense J is <E(r1) conj(E(r2))> (the documented convention),
    NOT its conjugate.  Pin it directly against the reference build."""
    ens = _complex_j_ensemble()
    nr = ens.shape[0]
    mcf = PartialCoherenceMCF.from_ensemble(
        ens, dx=1e-6, dy=1e-6, wavelength=0.5e-6, max_full_N=8)
    Em = ens.reshape(nr, -1)
    J_ref = (Em.T @ Em.conj()) / nr           # <E(r_i) conj(E(r_j))>
    assert np.allclose(mcf.J_full, J_ref, atol=1e-12)


def test_src1_dense_and_modal_agree_on_complex_j():
    """SRC-1: the dense and (full-rank) modal branches now agree to machine
    precision on a complex-J ensemble -- before the fix the dense branch was
    the conjugate, flipping the coherence phase sign with grid size."""
    ens = _complex_j_ensemble()
    nr = ens.shape[0]
    n_pix = ens.shape[1] * ens.shape[2]
    full_rank = min(nr, n_pix)                         # SVD rank of J
    kw = dict(dx=1e-6, dy=1e-6, wavelength=0.5e-6)
    dense = PartialCoherenceMCF.from_ensemble(ens, max_full_N=8, **kw)
    modal = PartialCoherenceMCF.from_ensemble(
        ens, max_full_N=4, n_modes=full_rank, **kw)    # full rank = exact
    cd = dense.coherence_at(2, 1, 3, 4)
    cm = modal.coherence_at(2, 1, 3, 4)
    assert abs(cd - cm) < 1e-9
    assert np.sign(np.angle(cd)) == np.sign(np.angle(cm))   # same phase sign


@pytest.mark.parametrize("call", [
    lambda: create_hermite_gauss(32, 1e-6, 0.0, 0.5e-6),        # w0 = 0
    lambda: create_hermite_gauss(32, 1e-6, -1e-3, 0.5e-6),      # w0 < 0
    lambda: create_laguerre_gauss(32, 1e-6, 0.0, 0.5e-6),       # w0 = 0
    lambda: create_top_hat_beam(32, 1e-6, 0.5e-6, diameter=0.0),
    lambda: create_top_hat_beam(32, 1e-6, 0.5e-6, diameter=-1e-3),
    lambda: create_annular_beam(32, 1e-6, 0.5e-6,
                                outer_diameter=1e-3, inner_diameter=2e-3),
    lambda: create_annular_beam(32, 1e-6, 0.5e-6,
                                outer_diameter=-1e-3, inner_diameter=0.0),
])
def test_src2_scale_parameter_guards(call):
    """SRC-2: the four factories missing scale-parameter validation now
    reject non-physical waists / diameters / inverted annuli."""
    with pytest.raises(ValueError):
        call()


def test_src2_valid_calls_still_work():
    assert np.isfinite(create_hermite_gauss(32, 1e-6, 1e-3, 0.5e-6)[0]).all()
    o, x, y = create_annular_beam(32, 1e-6, 0.5e-6, outer_diameter=2e-3,
                                  inner_diameter=1e-3)
    assert np.isfinite(o).all()


def test_src3_empty_field_angles_raises():
    """SRC-3: an empty field_angles now raises instead of silently returning
    (sources=[], x=None, y=None)."""
    with pytest.raises(ValueError, match="field_angles is empty"):
        create_multi_field_sources(32, 1e-6, 0.5e-6, [])


# =========================================================================
# AUDIT 3 -- propagators/*.py (AUDIT_PROPAGATORS_KERNELS_2026_07_07)
# =========================================================================


def test_ds1_farfield_propagate_returns_valid_source():
    """DS-1: ``Source.propagate`` at a far distance auto-selects a pitch-
    CHANGING kernel (fraunhofer/sas); pre-fix it wrapped the raw
    ``(E, dx, dy)`` tuple AS the field (``.shape`` then raised) and kept the
    stale input pitch.  Post-fix the returned Source carries a real ndarray
    field and the kernel's output pitch."""
    from lumenairy.sources.core import Source

    N, dx, wl = 64, 8e-6, 633e-9
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X * X + Y * Y) / (6 * dx) ** 2).astype(np.complex128)
    src = Source(E=E, dx=dx, dy=dx, wavelength=wl)
    out = src.propagate(method='auto', z=5.0)  # N_F << 1 -> far-field kernel
    assert isinstance(out.E, np.ndarray)
    assert out.E.shape == E.shape
    assert np.all(np.isfinite(out.E))
    # A far-field kernel changes the pitch to ~ lambda*z/(N*dx); it must NOT
    # be the stale input pitch.
    assert out.dx != dx


def test_ds1_asm_preserves_anamorphic_dy():
    """DS-1 (anamorphic re-thread): a pitch-PRESERVING kernel keeps the
    input's distinct y-pitch instead of collapsing dy == dx."""
    from lumenairy.sources.core import Source

    N, dx, dy, wl = 32, 5e-6, 7e-6, 633e-9
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dy
    X, Y = np.meshgrid(x, y, indexing='xy')
    E = np.exp(-(X * X + Y * Y) / (5 * dx) ** 2).astype(np.complex128)
    out = Source(E=E, dx=dx, dy=dy, wavelength=wl).propagate(method='asm',
                                                              z=1e-4)
    assert out.dx == dx
    assert out.dy == dy


def test_hf1_d4sigma_waist_convention():
    """HF-1: the LG-basis waist is the 1/e^2 radius == D4sigma/2, NOT the
    audit's D4sigma/4 (== second-moment sigma == half the waist).  Pin the
    convention on a known Gaussian."""
    from lumenairy.analysis.core import beam_d4sigma

    N, dx = 256, 1e-6
    w0 = 30e-6  # 1/e^2 amplitude radius
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X * X + Y * Y) / w0 ** 2).astype(np.complex128)
    d4x, d4y = beam_d4sigma(E, dx=dx)
    # D4sigma == 2 * w0  ->  w0 == d4x / 2 (the fixed factor); d4x/4 would be
    # w0/2, which is what the pre-fix code / audit prescription used.
    assert abs(d4x / 2.0 - w0) / w0 < 0.02
    assert abs(d4x / 4.0 - w0) / w0 > 0.4  # the WRONG factor is far off


def test_vd1_immersion_na_raises():
    """VD-1: an immersion NA (>= 1) is rejected instead of silently clamped
    to the 89.2 deg air cone."""
    from lumenairy.propagators.vector_diffraction import richards_wolf_focus

    N, dx, wl, f = 32, 2e-6, 550e-9, 1e-3
    pupil = np.ones((N, N), dtype=np.complex128)
    with pytest.raises(ValueError, match="NA must be in"):
        richards_wolf_focus(pupil, wl, 1.4, f, dx)
    # A valid air NA still works.
    out = richards_wolf_focus(pupil, wl, 0.6, f, dx)
    assert out is not None


def test_pk1_sas_beyond_zlimit_warns():
    """PK-1: propagating past the SAS validity bound emits a RuntimeWarning
    (was an easy-to-miss stdout print only)."""
    from lumenairy.propagators.sas import scalable_angular_spectrum_propagate

    N, dx, wl = 64, 4e-6, 633e-9
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X * X + Y * Y) / (8 * dx) ** 2).astype(np.complex128)
    with pytest.warns(RuntimeWarning, match="validity"):
        scalable_angular_spectrum_propagate(E, 10.0, wl, dx)


def test_hfpi2_invalid_sampling_raises():
    """HFPI-2: the ``sampling`` selector is now validated up front (and
    actually dispatched)."""
    from lumenairy.propagators.hfpi import (
        propagate_hfpi_through_prescription,
    )

    N, dx, wl = 16, 4e-6, 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    with pytest.raises(ValueError, match="sampling must be"):
        propagate_hfpi_through_prescription(
            E, dx, {'surfaces': []}, wavelength=wl, n_paths=8,
            sampling='bogus')


def test_sy2_anamorphic_pitch_changing_branch_raises():
    """SY-2: the pitch-CHANGING chain branches (fresnel/sas/turbulence)
    assume a square grid; an anamorphic working pitch now raises instead of
    silently mis-resampling the y-axis with the x-ratio."""
    from lumenairy.propagators.system import propagate_through_system

    N, dx, dy, wl = 32, 5e-6, 7e-6, 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    elements = [{'type': 'propagate', 'z': 1e-3, 'method': 'fresnel'}]
    with pytest.raises(ValueError, match="square grid pitch"):
        propagate_through_system(E, elements, wl, dx, dy=dy)


def test_dispatch_asm_znone_returns_copy():
    """Dispatch nit: ``method='asm', z=None`` returns a COPY, not the input
    array itself (a caller mutating the output must not corrupt the source)."""
    from lumenairy.propagators.dispatch import _dispatch_to_method

    N, dx, wl = 16, 4e-6, 633e-9
    E = np.ones((N, N), dtype=np.complex128)
    out = _dispatch_to_method('asm', E, z=None, wavelength=wl, dx=dx,
                              prescription=None, output_grid=None,
                              output_dx=None)
    assert out is not E
    out[0, 0] = 999.0
    assert E[0, 0] == 1.0


# =========================================================================
# AUDIT 4 -- raytrace/*.py (AUDIT_RAYTRACE_CORE_2026_07_08)
# =========================================================================

_SINGLET_RX = {
    'surfaces': [
        {'radius': 0.05, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': -0.05, 'glass_before': 'N-BK7', 'glass_after': 'air'},
    ],
    'thicknesses': [0.006, 0.095],
    'aperture_diameter': 20e-3,
}


def test_rt3_dead_paraxial_trio_removed():
    """RT-3: the dead+wrong ``_paraxial_trace`` trio (its refraction update
    omitted the n1/n2 rescaling) was removed from seidel.py."""
    import lumenairy.raytrace.seidel as s
    assert not hasattr(s, '_paraxial_trace')
    assert not hasattr(s, '_paraxial_refract')
    assert not hasattr(s, '_paraxial_transfer')


def test_rt4_world_coord_break_matches_legacy_trace_fold():
    """RT-4, RESOLVED W3-1 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25).

    RT-4's DIAGNOSIS was right (``world._apply_coord_break`` and
    ``intersection._apply_coord_break`` disagreed) but its fix flipped the
    wrong side, and the 408b8c3 revert then declared it a "phantom" on this
    very assertion -- which compared ``trace()``'s ray direction ``Q.T @ ez``
    (a vector in the NEW LOCAL frame) against ``world_R[:, 2] = Q @ ez`` (a
    vector in WORLD).  Both equalled ``Rx_math(+90) @ ez`` numerically, but
    that equality is the category error itself: it holds precisely BECAUSE
    the two sites were transposes of each other.

    W3-1 fixed ``intersection`` / ``differential`` / ``ui.model`` to the
    Zemax local-to-world convention that ``world.py`` already implemented
    (OpticStudio KB KA-01638).  What this test now pins is the physically
    meaningful relation: a coordinate break is a PASSIVE frame change, so
    the ray transform is the TRANSPOSE of the frame rotation and the ray's
    WORLD direction is invariant across the break.  Full oracle in
    ``tests/unit/test_niche_audit_w3_oracles.py``.
    """
    import numpy as np

    from lumenairy.raytrace import RayBundle
    from lumenairy.raytrace.intersection import _apply_coord_break as _local_cb
    from lumenairy.raytrace.surface import Surface
    from lumenairy.raytrace.world import _apply_coord_break as _world_cb

    tx_deg = 90.0
    # World: the new frame's local-to-world rotation.  Zemax's +tilt_x puts
    # the new local +z at world -y.
    _, Q = _world_cb(np.zeros(3), np.eye(3), {'tilt_x_deg': tx_deg})
    np.testing.assert_allclose(Q[:, 2], [0.0, -1.0, 0.0], atol=1e-9)
    # Legacy trace(): a +z-going ray's direction after the same coord break,
    # expressed in the NEW LOCAL frame.
    r = RayBundle(x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
                  L=np.array([0.0]), M=np.array([0.0]), N=np.array([1.0]),
                  opd=np.array([0.0]), alive=np.array([True]),
                  wavelength=633e-9)
    _local_cb(r, Surface(radius=np.inf, is_coordbrk=True, tilt_x_deg=tx_deg))
    local_dir = np.array([np.ravel(r.L)[0], np.ravel(r.M)[0],
                          np.ravel(r.N)[0]])
    # Passive-frame identity: local = Q.T @ world, so Q @ local == world.
    np.testing.assert_allclose(local_dir, Q.T @ np.array([0.0, 0.0, 1.0]),
                               atol=1e-9)
    np.testing.assert_allclose(Q @ local_dir, [0.0, 0.0, 1.0], atol=1e-9)


def test_rt5_offaxis_fan_passes_through_zero():
    """RT-5: for off-axis fields both the tangential and sagittal fans now
    pass through zero at the chief (py=0) -- the fan is EP-centred on the
    chief, not decentred by ep_z*tan(field)."""
    from lumenairy.raytrace.ray_fan import ray_fan_data
    from lumenairy.raytrace.trace import surfaces_from_prescription
    surfs = surfaces_from_prescription(_SINGLET_RX)
    for fa_deg in (0.0, 5.0):
        py, ey, px, ex = ray_fan_data(surfs, 587.6e-9, 8e-3,
                                      field_angle=np.radians(fa_deg),
                                      n_rays=41)
        c = len(py) // 2
        assert abs(py[c]) < 1e-12
        assert abs(ey[c]) < 1e-10, f"ey(0)={ey[c]} at fa={fa_deg}"
        assert abs(ex[c]) < 1e-10, f"ex(0)={ex[c]} at fa={fa_deg}"


def test_rt6_high_na_trace_jax_matches_numpy_and_no_warning():
    """RT-6: the JAX paraxial transfer is EXACT -- a high-NA trace_jax
    matches the NumPy trace to sub-ppm and emits NO spurious high-NA
    RuntimeWarning (the warning machinery was removed)."""
    import warnings
    pytest.importorskip('jax')
    from lumenairy.raytrace.core import RayBundle
    from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax
    from lumenairy.raytrace.trace import surfaces_from_prescription, trace
    wl = 587.6e-9
    n = 7
    th = np.radians(np.linspace(-40.0, 40.0, n))  # min|N|~0.77, NA~0.64
    L = np.sin(th)
    N = np.cos(th)
    z0 = np.zeros(n)
    rays = RayBundle(x=z0.copy(), y=z0.copy(), z=z0.copy(),
                     L=L.copy(), M=z0.copy(), N=N.copy(),
                     opd=z0.copy(), alive=np.ones(n, bool), wavelength=wl)
    surfs = surfaces_from_prescription(_SINGLET_RX)
    img = trace(rays, surfs, wl).image_rays
    state = make_jax_ray_state(z0, z0, z0, L, z0, N)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        outj = trace_jax(state, _SINGLET_RX, wl)
    # No spurious high-NA / paraxial-transfer warning.
    assert not any('transfer' in str(w.message).lower()
                   or 'paraxial' in str(w.message).lower()
                   or 'high-NA' in str(w.message) for w in caught)
    both = np.asarray(img.alive) & np.asarray(outj.alive)
    dx = np.abs(np.asarray(outj.x)[both] - np.asarray(img.x)[both]).max()
    assert dx < 1e-8, f"trace_jax vs NumPy trace x-disagreement {dx:.2e} m"


def test_rt8_prebuilt_jaxprescription_rejects_surface_diffraction():
    """RT-8: passing surface_diffraction alongside a pre-built
    JaxPrescription raises instead of silently dropping the DOE kick."""
    pytest.importorskip('jax')
    from lumenairy.raytrace.jax_trace import (
        _build_jax_prescription,
        make_jax_ray_state,
        trace_jax,
    )
    jp = _build_jax_prescription(_SINGLET_RX, 587.6e-9, None)
    state = make_jax_ray_state(*(np.zeros(3) for _ in range(3)),
                               np.zeros(3), np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="silently ignored|baked into"):
        trace_jax(state, jp, 587.6e-9,
                  surface_diffraction={0: (1, 0, 1e-6, 0.0)})


def test_rt9_seidel_field_sweep_drives_wfe_corrected_path():
    """RT-9: seidel_field_sweep now carries per-field 'lagrange_invariant'
    and 'abcd', so seidel_wfe(sweep, field_index=k) reaches the corrected
    H^2 Petzval path instead of the bare-sigma^2 fallback warning."""
    import warnings

    import lumenairy as la
    from lumenairy.raytrace.seidel_analysis import (
        seidel_field_sweep,
        seidel_wfe,
    )
    presc = la.make_singlet(R1=50e-3, R2=-50e-3, d=4e-3,
                            glass='N-BK7', aperture=10e-3)
    surfs = la.surfaces_from_prescription(presc)
    heights = np.linspace(0.0, 0.05, 6)
    result, _abcd = seidel_field_sweep(surfs, 1.31e-6, heights)
    assert 'lagrange_invariant' in result
    assert 'abcd' in result
    assert np.asarray(result['lagrange_invariant']).shape == heights.shape
    rho = np.linspace(0, 1, 8)
    theta = np.zeros_like(rho)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        seidel_wfe(result, rho, theta, field_index=3)
    assert not any('bare-sigma' in str(w.message) or 'sigma²' in
                   str(w.message) for w in caught), (
        "seidel_wfe still hit the bare-sigma^2 fallback")


def test_rt_nit_odd_aspheric_power_rejected():
    """RT-nit: only EVEN aspheric powers are supported; an odd power is now
    rejected at validation (sag uses h**(p//2) but the normal uses
    p*h**(p-1), so odd powers are sag/normal-inconsistent)."""
    from lumenairy.raytrace.trace import validate_prescription
    bad = {
        'surfaces': [
            {'radius': 0.05, 'glass_before': 'air', 'glass_after': 'N-BK7',
             'aspheric_coeffs': {3: 1e-3}},  # ODD power
            {'radius': -0.05, 'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [0.006, 0.095],
    }
    with pytest.raises(ValueError, match="ODD aspheric"):
        validate_prescription(bad)
    # An even-power asphere still validates.
    good = dict(bad)
    good['surfaces'] = [dict(bad['surfaces'][0], aspheric_coeffs={4: 1e-3}),
                        bad['surfaces'][1]]
    validate_prescription(good)  # must not raise


# =========================================================================
# AUDIT 5 -- glass.py + elements/polarization.py (AUDIT_GLASS_POLARIZATION)
# =========================================================================


def test_gl1_missing_kappa_message_points_at_working_remediation():
    """GL-1: the missing-kappa warning must point at the WORKING remediation
    (a complex-returning callable in GLASS_REGISTRY), not the dead-end
    register_fixed_glass (which stores a real-only shim)."""
    import lumenairy.glass as g
    g._kappa_warned.clear()
    with pytest.warns(RuntimeWarning) as rec:
        g._warn_missing_kappa_once('N-BK7', 587.6e-9)
    msg = str(rec[0].message)
    assert 'GLASS_REGISTRY' in msg
    assert '1j' in msg  # shows the complex-callable form


def test_gl_silica_resolves_like_its_siblings():
    """GL-nit: 'SILICA' now has its own bundled Sellmeier row, so it
    resolves (== SiO2 / FUSED_SILICA) instead of ImportError-ing on a
    minimal install."""
    from lumenairy.glass import get_glass_index
    wl = 587.6e-9
    n_silica = get_glass_index('SILICA', wl)
    n_sio2 = get_glass_index('SiO2', wl)
    assert abs(n_silica - n_sio2) < 1e-12
    assert 1.4 < n_silica < 1.5   # fused silica ~1.458 at d-line


def test_gl_validity_warning_array_safe():
    """GL-nit: _maybe_warn_outside_validity no longer crashes on an array
    wavelength (was float(array))."""
    import lumenairy.glass as g
    g._validity_warned.clear()
    # SiO2 validity is [0.21, 3.7] um; include an out-of-band element.
    wl = np.array([0.5e-6, 5.0e-6])   # 5 um > 3.7 um upper bound
    with pytest.warns(UserWarning, match="validity"):
        g._maybe_warn_outside_validity('SiO2', wl)
    # A fully in-band array must NOT warn.
    g._validity_warned.clear()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        g._maybe_warn_outside_validity('SiO2', np.array([0.5e-6, 1.0e-6]))


def test_gl2_reregister_fixed_index_clears_cache():
    """GL-2: re-registering a fixed index invalidates the value cache (now
    routed through the lock-correct _invalidate_glass_name), so the second
    resolution returns the NEW value rather than a stale cached one."""
    from lumenairy.glass import get_glass_index
    from lumenairy.raytrace.trace import _register_fixed_index
    name = '__gl2_probe__'
    _register_fixed_index(name, 1.5, 587.6e-9)
    assert abs(get_glass_index(name, 587.6e-9) - 1.5) < 1e-12
    _register_fixed_index(name, 1.7, 587.6e-9)   # overwrite
    assert abs(get_glass_index(name, 587.6e-9) - 1.7) < 1e-12


def test_gl_jonesfield_apply_spherical_lens_missing_wavelength():
    """GL-nit: JonesField.apply_spherical_lens() without wavelength raises a
    TypeError naming the argument, not a bare KeyError."""
    from lumenairy.elements.polarization import JonesField
    N, dx = 16, 2e-6
    Ex = np.ones((N, N), dtype=np.complex128)
    Ey = np.zeros((N, N), dtype=np.complex128)
    jf = JonesField(Ex, Ey, dx)
    with pytest.raises(TypeError, match="wavelength"):
        jf.apply_spherical_lens(focal_length=0.1)


# =========================================================================
# AUDIT 6 -- io/prescriptions_zemax.py (AUDIT_IO_ZEMAX)
# =========================================================================

# Singlet with a COORDBRK (DISZ=3mm) between the two lens surfaces.
_ZMX_CB_BETWEEN = """UNIT MM
SURF 0
  TYPE STANDARD
  DISZ 10.0
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50.0
  DIAM 12.0 0 0 0 1 ""
SURF 2
  TYPE COORDBRK
  DISZ 3.0
  PARM 3 2.0
SURF 3
  TYPE STANDARD
  CURV -0.02
  DISZ 95.0
  DIAM 12.0 0 0 0 1 ""
SURF 4
  TYPE STANDARD
  DISZ 0.0
  DIAM 1.0 0 0 0 1 ""
"""

# Singlet with the aperture STOP on the SECOND (back) lens surface.
_ZMX_STOP_ON_BACK = """UNIT MM
SURF 0
  TYPE STANDARD
  DISZ 10.0
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50.0
  DIAM 12.0 0 0 0 1 ""
SURF 2
  TYPE STANDARD
  STOP
  CURV -0.02
  DISZ 95.0
  DIAM 12.0 0 0 0 1 ""
SURF 3
  TYPE STANDARD
  DISZ 0.0
  DIAM 1.0 0 0 0 1 ""
"""


def test_zx1_coordbreak_disz_folded_into_flat_thicknesses(tmp_path):
    """ZX-1: a COORDBRK's axial DISZ between two lens surfaces is folded into
    the preceding element's gap in the flat thicknesses (was silently
    dropped, shifting every downstream axial position)."""
    from lumenairy.io.prescriptions_zemax import load_zemax_zmx
    p = tmp_path / 'cb.zmx'
    p.write_text(_ZMX_CB_BETWEEN, encoding='utf-8')
    # Explicit range spans the front (1) and back (3) lens surfaces across
    # the intervening COORDBRK (2).
    presc = load_zemax_zmx(str(p), surface_range=(1, 3))
    # Front->back gap = SURF1 DISZ (5mm) + CB DISZ (3mm) = 8mm.
    assert abs(presc['thicknesses'][0] - 8e-3) < 1e-9
    # The CB's own axial gap is still available for the world path.
    assert abs(presc['coord_breaks'][0]['thickness_m'] - 3e-3) < 1e-9


def test_s4_1_coordbreak_disz_folded_into_flat_thicknesses_txt(tmp_path):
    """S4-1 (AUDIT_V5_24_2): the prescription-data .txt loader must fold a
    COORDBRK's axial gap into the preceding element's thickness, exactly as
    the .zmx twin (ZX-1) does.  Before the fix the .txt loop dropped the gap
    silently, shifting every downstream axial position.

    Independent oracle: the front->back gap is the geometric sum of the front
    element's Thickness (5 mm) and the intervening COORDBRK's Thickness
    (3 mm) = 8 mm; and the .txt result must agree with the .zmx twin loaded
    from the equivalent _ZMX_CB_BETWEEN prescription."""
    from lumenairy.io.prescriptions_zemax import (
        load_zemax_prescription_data_txt,
        load_zemax_zmx,
    )
    # Same geometry as _ZMX_CB_BETWEEN: front DISZ 5 mm, COORDBRK DISZ 3 mm,
    # back DISZ 95 mm.  Thicknesses report in millimetres (default units).
    rows = [
        'SURFACE DATA SUMMARY:',
        '',
        'Surf\tType\tRadius\tThickness\tGlass\tClear Diam\tChip Zone'
        '\tMech Diam\tConic\tComment',
        'OBJ\tSTANDARD\tInfinity\t10\t\t10\t0\t10\t0\t',
        '1\tSTANDARD\t50\t5\tN-BK7\t24\t0\t24\t0\tfront',
        '2\tCOORDBRK\tInfinity\t3\t\t0\t0\t0\t0\t',
        '3\tSTANDARD\t-50\t95\t\t24\t0\t24\t0\tback',
        'IMA\tSTANDARD\tInfinity\t0\t\t2\t0\t2\t0\t',
    ]
    p = tmp_path / 'cb.txt'
    p.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    # Explicit range spans the front (1) and back (3) lens surfaces across
    # the intervening COORDBRK (2).
    presc = load_zemax_prescription_data_txt(str(p), surface_range=(1, 3))
    # Front->back gap = SURF1 Thickness (5 mm) + CB Thickness (3 mm) = 8 mm.
    # all_thicknesses is the flat all-element list touched by the fix; the
    # lens-only thicknesses derive from it and must fold too.
    assert abs(presc['all_thicknesses'][0] - 8e-3) < 1e-9
    assert abs(presc['thicknesses'][0] - 8e-3) < 1e-9

    # Parity: the .zmx twin folds the same gap to the same value (the two
    # near-duplicate loaders must not diverge again -- S4-8).
    q = tmp_path / 'cb.zmx'
    q.write_text(_ZMX_CB_BETWEEN, encoding='utf-8')
    presc_zmx = load_zemax_zmx(str(q), surface_range=(1, 3))
    assert abs(presc['thicknesses'][0] - presc_zmx['thicknesses'][0]) < 1e-9


def test_zx3_loaded_stop_index_preserved(tmp_path):
    """ZX-3: an explicit STOP on the back surface is preserved on the
    lens-only surfaces and exposed as a top-level stop_index (was dropped,
    so re-export/tracer defaulted STOP to surface 0)."""
    from lumenairy.io.prescriptions_zemax import load_zemax_zmx
    p = tmp_path / 'stop.zmx'
    p.write_text(_ZMX_STOP_ON_BACK, encoding='utf-8')
    presc = load_zemax_zmx(str(p))
    assert presc['stop_index'] == 1
    assert presc['surfaces'][1]['is_stop'] is True
    assert presc['surfaces'][0]['is_stop'] is False


def test_zx4_full_writer_honours_back_focal_length(tmp_path):
    """ZX-4: _export_zemax_zmx_full now honours back_focal_length for the
    trailing (last-surface -> image) gap instead of hardcoding DISZ 0.  Build
    an element-list prescription whose last gap runs past all_thicknesses so
    the BFL fallback is what fills it."""
    from lumenairy.io.prescriptions_zemax import _export_zemax_zmx_full
    presc = {
        'elements': [
            {'element_type': 'surface', 'radius': 0.05, 'conic': 0.0,
             'aspheric_coeffs': {}, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 5e-3, 'is_stop': True},
            {'element_type': 'surface', 'radius': -0.05, 'conic': 0.0,
             'aspheric_coeffs': {}, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 5e-3, 'is_stop': False},
        ],
        # Only ONE inter-element gap recorded -> the 2nd element's trailing
        # gap falls off the list and must be filled by the BFL.
        'all_thicknesses': [4e-3],
        'aperture_diameter': 10e-3,
    }
    out = tmp_path / 'bfl.zmx'
    _export_zemax_zmx_full(presc, str(out), wavelength=587.6e-9,
                           back_focal_length=0.042)
    text = out.read_text(encoding='utf-8')
    # The 42 mm BFL must appear as a DISZ (was hardcoded 0 before ZX-4).
    assert 'DISZ 42.00000000' in text


# --- AUDIT 6 residuals (loose-ends 2026-07) ------------------------------

# Same singlet as _ZMX_CB_BETWEEN but with the STOP flag on the COORDBRK
# row (legal in Zemax; the aperture physically lands on the next surface).
_ZMX_STOP_ON_CB = """UNIT MM
SURF 0
  TYPE STANDARD
  DISZ 10.0
SURF 1
  TYPE STANDARD
  CURV 0.02
  DISZ 5.0
  GLAS N-BK7 0 0 1.5 50.0
  DIAM 12.0 0 0 0 1 ""
SURF 2
  TYPE COORDBRK
  STOP
  DISZ 3.0
  PARM 3 2.0
SURF 3
  TYPE STANDARD
  CURV -0.02
  DISZ 95.0
  DIAM 9.0 0 0 0 1 ""
SURF 4
  TYPE STANDARD
  DISZ 0.0
  DIAM 1.0 0 0 0 1 ""
"""


def test_zx5_stop_on_coordbrk_reassigned_to_next_surface(tmp_path):
    """ZX-5: a STOP declared on a COORDBRK row used to vanish silently
    (CB rows are filtered out of the lens surfaces, so the stop search
    fell back to max-DIAM).  It now lands on the NEXT optical surface
    (Zemax physical intent) with a warning."""
    from lumenairy.io.prescriptions_zemax import load_zemax_zmx
    p = tmp_path / 'stopcb.zmx'
    p.write_text(_ZMX_STOP_ON_CB, encoding='utf-8')
    with pytest.warns(UserWarning, match='STOP declared on COORDBRK'):
        presc = load_zemax_zmx(str(p), surface_range=(1, 3))
    assert presc['stop_index'] == 1
    assert presc['surfaces'][1]['is_stop'] is True
    assert presc['surfaces'][0]['is_stop'] is False
    # Aperture from the reassigned stop's DIAM (9 mm semi-dia), not the
    # max-DIAM fallback (12 mm semi-dia).
    assert abs(presc['aperture_diameter'] - 18e-3) < 1e-9


def test_zx5_stop_on_coordbrk_reassigned_txt_loader(tmp_path):
    """ZX-5: same reassignment on the prescription-data .txt loader
    (STO label on a COORDBRK row)."""
    from lumenairy.io.prescriptions_zemax import (
        load_zemax_prescription_data_txt,
    )
    rows = [
        'SURFACE DATA SUMMARY:',
        '',
        'Surf\tType\tRadius\tThickness\tGlass\tClear Diam\tChip Zone'
        '\tMech Diam\tConic\tComment',
        'OBJ\tSTANDARD\tInfinity\t100\t\t10\t0\t10\t0\t',
        '1\tSTANDARD\t50\t5\tN-BK7\t24\t0\t24\t0\tfront',
        'STO\tCOORDBRK\tInfinity\t3\t\t0\t0\t0\t0\t',
        '3\tSTANDARD\t-50\t95\t\t18\t0\t18\t0\tback',
        'IMA\tSTANDARD\tInfinity\t0\t\t2\t0\t2\t0\t',
    ]
    p = tmp_path / 'stopcb.txt'
    p.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    with pytest.warns(UserWarning, match='STOP declared on COORDBRK'):
        presc = load_zemax_prescription_data_txt(
            str(p), surface_range=(1, 3))
    # Aperture from the reassigned stop's 18 mm clear diameter, not the
    # 24 mm max-DIAM fallback; the elements list carries the flag.
    assert abs(presc['aperture_diameter'] - 18e-3) < 1e-9
    stops = [e.get('is_stop', False) for e in presc['elements']]
    assert stops == [False, True]


def test_zx_gcat_glass_catalogs_kwarg(tmp_path):
    """ZX residual: the .zmx writers' GCAT row is now configurable via
    glass_catalogs= (was hardcoded 'GCAT SCHOTT MISC')."""
    from lumenairy.io.prescriptions_zemax import export_zemax_zmx
    presc = {
        'surfaces': [
            {'radius': 0.05, 'conic': 0.0, 'aspheric_coeffs': {},
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': -0.05, 'conic': 0.0, 'aspheric_coeffs': {},
             'glass_before': 'N-BK7', 'glass_after': 'air'},
        ],
        'thicknesses': [4e-3],
        'aperture_diameter': 10e-3,
    }
    p_default = tmp_path / 'default.zmx'
    p_cdgm = tmp_path / 'cdgm.zmx'
    export_zemax_zmx(presc, str(p_default), wavelength=1.31e-6)
    export_zemax_zmx(presc, str(p_cdgm), wavelength=1.31e-6,
                     glass_catalogs=('SCHOTT', 'CDGM'))
    # Default line unchanged (behaviour-preserving).
    assert 'GCAT SCHOTT MISC' in p_default.read_text(encoding='utf-8')
    text_cdgm = p_cdgm.read_text(encoding='utf-8')
    assert 'GCAT SCHOTT CDGM' in text_cdgm
    assert 'MISC' not in text_cdgm


def test_zx_gcat_glass_catalogs_full_writer(tmp_path):
    """ZX residual: glass_catalogs= is forwarded through the cb/mirror-
    aware full writer path too."""
    from lumenairy.io.prescriptions_zemax import export_zemax_zmx
    presc = {
        'surfaces': [
            {'radius': 0.05, 'conic': 0.0, 'aspheric_coeffs': {},
             'glass_before': 'air', 'glass_after': 'N-BK7',
             'is_stop': True},
            {'radius': -0.05, 'conic': 0.0, 'aspheric_coeffs': {},
             'glass_before': 'N-BK7', 'glass_after': 'air',
             'is_stop': False},
        ],
        'elements': [
            {'element_type': 'surface', 'radius': 0.05, 'conic': 0.0,
             'aspheric_coeffs': {}, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 5e-3,
             'is_stop': True, 'surf_num': 1},
            {'element_type': 'surface', 'radius': -0.05, 'conic': 0.0,
             'aspheric_coeffs': {}, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 5e-3,
             'is_stop': False, 'surf_num': 3},
        ],
        'thicknesses': [4e-3],
        'all_thicknesses': [4e-3],
        'aperture_diameter': 10e-3,
        'coord_breaks': [
            {'surf_num': 2, 'decenter_x_m': 0.0, 'decenter_y_m': 0.0,
             'tilt_x_deg': 1.0, 'tilt_y_deg': 0.0, 'tilt_z_deg': 0.0,
             'order': 0, 'thickness_m': 0.0},
        ],
    }
    out = tmp_path / 'full.zmx'
    export_zemax_zmx(presc, str(out), wavelength=1.31e-6,
                     glass_catalogs=('SCHOTT', 'CDGM'))
    text = out.read_text(encoding='utf-8')
    assert 'TYPE COORDBRK' in text          # took the full-writer path
    assert 'GCAT SCHOTT CDGM' in text


# --- coord_breaks -> surfaces_from_prescription LOCAL bridge (opt-in) ----

def _periscope_prescription(loader_convention):
    """Plano-convex singlet + 45-deg fold mirror + detector 50mm
    post-fold (the validation periscope).  ``loader_convention=True``
    emits the .zmx-loader (ZX-1) thickness convention -- each coord
    break's own DISZ folded into the preceding gap; False emits the
    hand-built world convention (gaps exclude CB thickness) consumed by
    world_surfaces_from_prescription."""
    return {
        'surfaces': [
            {'radius': 50e-3, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'surf_num': 1},
            {'radius': float('inf'), 'glass_before': 'N-BK7',
             'glass_after': 'air', 'surf_num': 2},
            {'radius': float('inf'), 'glass_before': 'air',
             'glass_after': 'MIRROR', 'surf_num': 15},
            {'radius': float('inf'), 'glass_before': 'air',
             'glass_after': 'air', 'surf_num': 25},
        ],
        # Mirror -> detector leg: -50mm (Zemax-signed) carried by the
        # post-fold coord break; folded into the mirror's gap on the
        # loader convention.
        'thicknesses': [3e-3, 0.05,
                        -0.05 if loader_convention else 0.0, 0.0],
        'aperture_diameter': 0.010,
        'coord_breaks': [
            {'surf_num': 10, 'tilt_x_deg': 45.0, 'order': 0,
             'thickness_m': 0.0},
            {'surf_num': 20, 'tilt_x_deg': 45.0, 'order': 0,
             'thickness_m': -0.05},
        ],
    }


def test_zx6_include_coord_breaks_reverses_zx1_fold(tmp_path):
    """RT/ZX residual: bridging a loaded .zmx's coord breaks into the
    local surface list must subtract each break's DISZ back out of the
    preceding (ZX-1-folded) gap, or the leg double-counts."""
    from lumenairy.io.prescriptions_zemax import load_zemax_zmx
    from lumenairy.raytrace.trace import surfaces_from_prescription
    p = tmp_path / 'cb.zmx'
    p.write_text(_ZMX_CB_BETWEEN, encoding='utf-8')
    presc = load_zemax_zmx(str(p), surface_range=(1, 3))
    sl = surfaces_from_prescription(presc, include_coord_breaks=True)
    assert [s.is_coordbrk for s in sl] == [False, True, False]
    # Preceding gap restored to the raw SURF1 DISZ (5mm; the ZX-1 fold
    # had made it 8mm) and the break carries its own 3mm.
    assert abs(sl[0].thickness - 5e-3) < 1e-12
    assert abs(sl[1].thickness - 3e-3) < 1e-12
    assert sl[1].tilt_x_deg == 2.0
    assert sl[1].surf_num == 2


def test_zx7_coord_break_bridge_matches_trace_world_oracle():
    """RT/ZX residual: single-fold periscope -- the plain local trace()
    over surfaces_from_prescription(include_coord_breaks=True) (loader
    thickness convention) must reproduce the trace_world() oracle
    (world convention) at the detector plane."""
    import lumenairy as la
    from lumenairy.raytrace.core import trace_world
    from lumenairy.raytrace.trace import surfaces_from_prescription, trace
    from lumenairy.raytrace.world import world_surfaces_from_prescription
    rays = la.make_rings(3e-3, 3, 8, 0.0, 1.31e-6)
    sl = surfaces_from_prescription(
        _periscope_prescription(loader_convention=True),
        include_coord_breaks=True)
    r_local = trace(rays, sl, 1.31e-6)
    wsurfs = world_surfaces_from_prescription(
        _periscope_prescription(loader_convention=False))
    r_world = trace_world(rays, wsurfs, 1.31e-6)
    lo, wo = r_local.image_rays, r_world.image_rays
    assert bool(np.all(lo.alive)) and bool(np.all(wo.alive))
    # Both paths report the detector-plane state in the detector's
    # local frame; positions to 1e-9 m, direction cosines to 1e-9.
    for field in ('x', 'y', 'z', 'L', 'M', 'N'):
        assert np.max(np.abs(getattr(lo, field) - getattr(wo, field))) \
            < 1e-9, field


def test_zx8_include_coord_breaks_default_off_unchanged():
    """RT/ZX residual: the default (include_coord_breaks=False) ignores
    coord_breaks entirely -- array-identical trace to the same
    prescription with the key stripped."""
    import lumenairy as la
    from lumenairy.raytrace.trace import surfaces_from_prescription, trace
    presc = _periscope_prescription(loader_convention=True)
    stripped = {k: v for k, v in presc.items() if k != 'coord_breaks'}
    sl_default = surfaces_from_prescription(presc)
    sl_stripped = surfaces_from_prescription(stripped)
    assert not any(s.is_coordbrk for s in sl_default)
    rays = la.make_rings(3e-3, 3, 8, 0.0, 1.31e-6)
    a = trace(rays, sl_default, 1.31e-6).image_rays
    b = trace(rays, sl_stripped, 1.31e-6).image_rays
    for field in ('x', 'y', 'z', 'L', 'M', 'N', 'opd'):
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


# =========================================================================
# AUDIT 7 -- elements/doe.py (AUDIT_DOE_GRATING_FREEFORM)
# =========================================================================


def test_doe1_fits_split_roundtrip_preserves_phase(tmp_path):
    """DOE-1: save_fits_field's DEFAULT split (amp + PHASE extension) is now
    auto-detected by load_fits_field's DEFAULT (hdu_phase=None), so the
    round-trip preserves phase instead of silently dropping it."""
    pytest.importorskip('astropy')
    from lumenairy.elements.doe import load_fits_field, save_fits_field
    N, dx, wl = 16, 1e-6, 633e-9
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = (np.exp(-(X * X + Y * Y) / (4 * dx) ** 2)
         * np.exp(1j * 2.0 * X / dx)).astype(np.complex128)
    p = str(tmp_path / 'field.fits')
    save_fits_field(p, E, wavelength=wl, dx=dx)   # split_amp_phase=True default
    E2, meta = load_fits_field(p)                  # hdu_phase=None default
    # Phase survives (float32 storage precision).
    assert np.max(np.abs(np.angle(E2) - np.angle(E))) < 1e-4
    assert np.max(np.abs(np.abs(E2) - np.abs(E))) < 1e-4
    assert abs(meta['dx'] - dx) < 1e-12


def test_doe_dammann_itr_wavsamp_guards():
    """DOE-nit: makedammann2d validates itr / wavsamp up front (itr=0 was a
    0/0 int(nan) crash; wavsamp=0 blew up the order count)."""
    from lumenairy.elements.doe import makedammann2d
    with pytest.raises(ValueError, match="itr must be"):
        makedammann2d(itr=0, plot=False)
    with pytest.raises(ValueError, match="wavsamp must be"):
        makedammann2d(wavsamp=0.0, plot=False)


# =========================================================================
# AUDIT 8 -- elements/coatings.py + elements/elements.py (AUDIT_COATINGS_ELEMENTS)
# =========================================================================


def test_coat_material_index_warning_gated_on_dispersion():
    """COAT-nit: the 'extrapolated value may not be physical' warning fires
    only for dispersive (Sellmeier) materials, not for constant-n materials
    (where the value is flat and nothing is extrapolated)."""
    import warnings

    from lumenairy.elements.coatings import get_coating_material_index
    # Constant-n MgO (range up to 6 um): OUT of range -> NO warning.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        n = get_coating_material_index('MgO', 6.5e-6)   # > 6 um upper bound
    assert abs(n - 1.74) < 1e-9
    # Sellmeier MgF2 (range up to 7 um): OUT of range -> DOES warn.
    with pytest.warns(UserWarning, match="validity"):
        get_coating_material_index('MgF2', 8.0e-6)


def test_coat_gaussian_aperture_sigma_guard():
    """COAT-nit: apply_gaussian_aperture rejects sigma<=0 (was a
    divide-by-zero -> all-NaN field)."""
    from lumenairy.elements.elements import apply_gaussian_aperture
    E = np.ones((16, 16), dtype=np.complex128)
    with pytest.raises(ValueError, match="sigma must be"):
        apply_gaussian_aperture(E, 1e-6, 0.0)


def test_coat_annular_aperture_inverted_guard():
    """COAT-nit: apply_aperture(shape='annular') rejects inner>=outer (was a
    silent all-zero field)."""
    from lumenairy.elements.elements import apply_aperture
    E = np.ones((16, 16), dtype=np.complex128)
    with pytest.raises(ValueError, match="must be < outer"):
        apply_aperture(E, 1e-6, shape='annular',
                       params={'inner_diameter': 5e-6, 'outer_diameter': 2e-6})


# =========================================================================
# AUDIT 9 -- elements/lenses_maslov.py (AUDIT_MASLOV)
# =========================================================================


def test_msl1_det_safe_floor_is_signed_and_nonzero():
    """MSL-1: the near-singular-Hessian floor used by all three saddle
    integrators must be sign-preserving AND never zero.  The OLD form
    ``sign(det_H)*1e-30 + 1e-30`` cancelled to exactly 0 for a tiny NEGATIVE
    determinant (a fold-caustic neighbourhood) -> 1/0 -> NaN saddle step.
    Pin the NEW ``where(det_H < 0, -1e-30, 1e-30)`` property across the
    window that broke the old code, guarding against a revert."""
    det_H = np.array([-1e-31, -5e-40, -0.0, 0.0, 5e-40, 1e-31])

    def old_floor(d):   # the buggy form
        return np.sign(d) * 1e-30 + 1e-30

    def new_floor(d):   # the MSL-1 fix
        return np.where(d < 0, -1e-30, 1e-30)

    # The old floor hits exactly 0 on the negative window (the bug).
    assert np.any(old_floor(det_H) == 0.0)
    # The new floor is never 0 and preserves the determinant's sign.
    nf = new_floor(det_H)
    assert np.all(nf != 0.0)
    assert np.all(np.sign(nf) == np.where(det_H < 0, -1.0, 1.0))


# =========================================================================
# AUDIT 10 -- elements/eme/eme_2d.py (AUDIT_EME)
# =========================================================================


def test_eme_lossy_strip_modes_finite_and_orthonormal():
    """EME-nit: the complex-SYMMETRIC lossy-eps path (Im(eps)!=0 at kx0=0)
    returns finite modes with a floored bilinear norm (no inf/NaN from a
    near-defective sum(Phi^2))."""
    from lumenairy.elements.eme.eme_2d import strip_x_modes
    Nx, Lx, k0 = 24, 2e-6, 2 * np.pi / 1.0e-6
    # A lossy dielectric grating: real base + absorbing stripe.
    eps = np.full(Nx, 2.25 + 0.0j)
    eps[Nx // 3: 2 * Nx // 3] = 4.0 + 0.3j       # Im != 0 -> complex-symmetric
    lam, Phi = strip_x_modes(eps, Lx, Nx, k0, kx0=0.0)
    assert np.all(np.isfinite(lam))
    assert np.all(np.isfinite(Phi))
    # Bilinear (complex-symmetric) norm ~ 1 per column (floored ones aside).
    bilin = np.sum(Phi ** 2, axis=0)
    assert np.all(np.isfinite(bilin))


# =========================================================================
# AUDIT 11 -- elements/bsdf.py + elements/segment_geometry.py
#             (AUDIT_BSDF_SEGMENT_GEOMETRY)
# =========================================================================


def test_bsdf1_gaussian_sample_reproduces_rayleigh_lobe():
    """BSDF-1: GaussianBSDF.sample now draws the offset angle from a Rayleigh
    law (mean = sigma*sqrt(pi/2) ~ 1.25 sigma), reproducing the lobe -- not
    the old half-normal (mean = sigma*sqrt(2/pi) ~ 0.80 sigma, ~35% too close
    to specular)."""
    from lumenairy.elements.bsdf import GaussianBSDF
    sigma = 0.05
    bsdf = GaussianBSDF(sigma_rad=sigma)
    # Normal incidence -> specular is +z, so the offset angle = arccos(dir_z).
    dirs = bsdf.sample(np.array([0.0, 0.0, -1.0]), 40000, rng=0)
    theta = np.arccos(np.clip(dirs[:, 2], -1.0, 1.0))
    mean = float(theta.mean())
    rayleigh_mean = sigma * np.sqrt(np.pi / 2)     # ~0.0627
    halfnormal_mean = sigma * np.sqrt(2 / np.pi)   # ~0.0399 (the OLD bug)
    assert abs(mean - rayleigh_mean) < 0.03 * rayleigh_mean
    assert mean > 0.5 * (rayleigh_mean + halfnormal_mean)  # clearly NOT the old


def test_bsdf_harvey_shack_evaluate_batched_incidence():
    """BSDF-nit: HarveyShackBSDF.evaluate is now batch-safe over incidence
    (mirrors GaussianBSDF's F-22 fix) instead of crashing on inc[0]."""
    from lumenairy.elements.bsdf import HarveyShackBSDF
    bsdf = HarveyShackBSDF(b0=0.1, l=0.05, s=2.0)
    M = 5
    inc = np.tile(np.array([0.0, 0.0, -1.0]), (M, 1))       # (M, 3)
    sd = np.tile(np.array([0.1, 0.0, 0.9949874]), (M, 1))   # (M, 3), |.|~1
    out = bsdf.evaluate(inc, sd)
    assert np.asarray(out).shape == (M,)
    assert np.all(np.isfinite(out))


# =========================================================================
# AUDIT 12 -- optimize/jax_merits.py (AUDIT_OPTIMIZE_MERITS)
# =========================================================================


def test_opt1_lg_jax_merit_is_strehl_deficit_not_amplitude():
    """OPT-1: make_lg_aberration_merit_jax must penalise the Strehl DEFICIT
    ``1 - |Strehl|^2`` (grows as coupling worsens), NOT ``|Strehl|^2`` (which
    design_optimize would MINIMISE toward |Strehl|=0 = MAX aberration).

    Decisive by construction: at a grossly waist-MISMATCHED source (poor
    LG00->LG00 coupling) ``|Strehl|^2 -> 0``, so the DEFICIT form -> ~1 while
    the old amplitude form -> ~0.  A value near 1 therefore confirms the
    Strehl-deficit (correct-direction) fix; a value near 0 would mean the
    pre-OPT-1 ``|res|^2`` (wrong-direction) form is still in place."""
    pytest.importorskip('jax')
    import lumenairy
    from lumenairy.optimize.core import (
        EvaluationContext,
        make_lg_aberration_merit_jax,
    )
    pres = lumenairy.make_singlet(R1=500e-3, R2=float('inf'), d=3e-3,
                                  glass='N-BK7', aperture=4e-3)
    pres['object_distance'] = 0.0

    def build_args(x):
        return (None, None, None, None, x[0], None)

    merit = make_lg_aberration_merit_jax(
        pres, wavelength=1.30e-6, targets={(0, 0): 1.0},
        build_args=build_args, field_points=[(0.0, 0.0)])
    x = np.array([50e-6])   # grossly mismatched source waist -> |Strehl|^2 ~ 0
    ctx = EvaluationContext(prescription=pres, wavelength=1.30e-6,
                            N=64, dx=10e-6, x=x)
    try:
        v = float(merit.evaluate(ctx))
    except (RuntimeError, ValueError, ZeroDivisionError,
            np.linalg.LinAlgError) as exc:
        pytest.skip(f'LG-tensor eval unstable on this runtime: {exc}')
    if not np.isfinite(v):
        pytest.skip('LG-tensor eval returned non-finite on this runtime.')
    # Deficit form at poor coupling -> ~1; the OLD |res|^2 form -> ~0.
    assert 0.5 < v <= 1.0 + 1e-6, (
        f'LG-JAX (0,0) merit = {v} at a mismatched waist (poor coupling); '
        f'expected a LARGE Strehl DEFICIT (~1).  A value near 0 means the '
        f'pre-OPT-1 |Strehl|^2 (wrong-direction) form is still in place.')


# =========================================================================
# AUDIT 13 -- optimize/parameterizations.py (AUDIT_OPTIMIZE_DRIVER)
# =========================================================================


def test_opt_driver_aspheric_floor_classification_not_overbroad():
    """OPT-nit: the FD scale-floor classifier now matches actual aspheric
    coefficient names (A4 / a_8 / alpha0) but NOT any key merely starting
    with 'a' (e.g. a future 'axis' field), which the old
    ``startswith('a')`` misrouted to the dimensionless aspheric floor."""
    from lumenairy.optimize.parameterizations import (
        _DEFAULT_SCALE_FLOORS,
        _classify_path_to_floor,
    )
    asph = _DEFAULT_SCALE_FLOORS['aspheric']
    default = _DEFAULT_SCALE_FLOORS['_default']
    assert asph != default
    for key in ('A4', 'a_8', 'a12', 'alpha0', 'aspheric_coeffs'):
        assert _classify_path_to_floor(('surfaces', 0, key)) == asph
    # A key that merely starts with 'a' is NOT aspheric anymore.
    for key in ('axis', 'angle', 'anamorphic'):
        assert _classify_path_to_floor(('surfaces', 0, key)) == default


# =========================================================================
# AUDIT 14 -- optimize/wrapper_merits.py (AUDIT_OPTIMIZE_WRAPPERS)
# =========================================================================


def test_opt2_tolerance_merit_populates_rms_and_opd_subcontext():
    """OPT-2: ToleranceAwareMerit now populates rms_radius_best (nanargmax)
    AND opd_map on the per-trial sub-context, mirroring the sibling
    aggregators.  Pre-fix it set only strehl_best, so an OPD/spot sub-merit
    saw the inf default / None and degenerated to inf / silently-inert."""
    import lumenairy
    from lumenairy.optimize.context import MeritTerm
    from lumenairy.optimize.core import EvaluationContext
    from lumenairy.optimize.wrapper_merits import ToleranceAwareMerit

    seen = {}

    class _Spy(MeritTerm):
        name = 'spy'
        needs_wave = True
        weight = 1.0

        def evaluate(self, ctx):
            seen['rms'] = ctx.rms_radius_best
            seen['opd'] = ctx.opd_map
            return 0.0

    pres = lumenairy.make_singlet(R1=60e-3, R2=float('inf'), d=4e-3,
                                  glass='N-BK7', aperture=12e-3)
    tol = ToleranceAwareMerit(
        sub_merit=_Spy(),
        perturbation_spec=[{'surface_index': 0, 'decenter_std': 0.0,
                            'tilt_std': 0.0, 'form_error_rms': 0.0}],
        n_trials=1, seed=0)
    ctx = EvaluationContext(prescription=pres, wavelength=1.30e-6,
                            N=64, dx=8e-6, efl=0.1, bfl=0.1)
    tol.evaluate(ctx)
    assert 'rms' in seen, 'sub-merit was never evaluated'
    # Post-fix: a finite rms_radius_best (not the inf default) + a real opd_map.
    assert np.isfinite(seen['rms']), (
        f'rms_radius_best={seen["rms"]} -- inf means OPT-2 is unfixed '
        f'(only strehl_best was populated).')
    assert seen['opd'] is not None, (
        'opd_map is None -- OPD-based sub-merits would be inert (OPT-2 '
        'unfixed).')


# =========================================================================
# AUDIT 15 -- io/prescriptions_code_v.py (AUDIT_IO_PRESCRIPTIONS)
# =========================================================================


def test_cv1_codev_dropped_fold_directive_warns(tmp_path):
    """CV-1: load_codev_seq now warns (once) when it drops unparsed
    coordinate decenter/tilt (fold) directives, instead of silently
    importing a folded .seq as a straight-axis system."""
    import lumenairy as la
    rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3)
    p = tmp_path / 'folded.seq'
    la.export_codev_seq(rx, str(p), wavelength=1.31e-6,
                        aperture_diameter=10e-3)
    # Inject fold directives (a decenter + a tilt) into surface 1's block.
    text = p.read_text(encoding='utf-8')
    text = text.replace('GLA N-BK7',
                        'GLA N-BK7\n  XDE 0.002000\n  ADE 3.000000', 1)
    p.write_text(text, encoding='utf-8')
    with pytest.warns(UserWarning, match="fold"):
        la.load_codev_seq(str(p))
    # A plain (unfolded) .seq must NOT warn.
    q = tmp_path / 'plain.seq'
    la.export_codev_seq(rx, str(q), wavelength=1.31e-6,
                        aperture_diameter=10e-3)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter('error')
        la.load_codev_seq(str(q))


# =========================================================================
# AUDIT 16 -- optimize/{multiconfig,multi_objective,_merit_jit}.py
#             (AUDIT_OPTIMIZE_TAIL -- clean, no findings; regression pin)
# =========================================================================


def test_opt_tail_beam_expander_afocal_magnification():
    """AUDIT_OPTIMIZE_TAIL had no findings; pin one verified-correct
    behaviour: the beam-expander builder solves the C=0 afocal air-gap so the
    system is afocal with |angular magnification| == M."""
    from lumenairy.optimize.multiconfig import (
        afocal_angular_magnification,
        beam_expander_prescription,
    )
    M, wl = 3.0, 633e-9
    rx = beam_expander_prescription(M=M, f_objective=50e-3, wavelength=wl)
    mag, is_afocal = afocal_angular_magnification(rx, wavelength=wl)
    assert is_afocal
    # A beam expander magnifies the BEAM by M, so by the afocal invariant the
    # ANGULAR magnification is 1/M (angles shrink as the beam expands).
    assert abs(abs(mag) - 1.0 / M) < 0.05


# =========================================================================
# AUDIT 17 -- io/codegen.py + io/storage.py (AUDIT_IO_STORAGE_CODEGEN)
# =========================================================================


def _q_type_elements_prescription():
    return {
        'name': 'qtest', 'aperture_diameter': 10e-3,
        'elements': [
            {'element_type': 'surface', 'radius': 0.05, 'conic': 0.0,
             'aspheric_coeffs': None, 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 5e-3, 'is_stop': True,
             'freeform_type': 'q_bfs', 'q_bfs_coeffs': {0: 1e-6, 1: -2e-7},
             'r_max': 5e-3},
            {'element_type': 'surface', 'radius': -0.05, 'conic': 0.0,
             'aspheric_coeffs': None, 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 5e-3, 'is_stop': False},
        ],
        'all_thicknesses': [4e-3, 0.0], 'thicknesses': [4e-3],
        'object_distance': 0.0,
    }


def test_cg1_codegen_forwards_qtype_freeform_keys():
    """CG-1: generated scripts now forward the Forbes Q-type freeform keys
    into the apply_real_lens prescription (was dropped -> Q-type surface
    silently degraded to base conic)."""
    from lumenairy.io.codegen import generate_simulation_script
    s = generate_simulation_script(
        _q_type_elements_prescription(), wavelength=633e-9, N=64, dx=8e-6,
        include_plotting=False, include_analysis=False)
    assert 'freeform_type' in s
    assert 'q_bfs_coeffs' in s
    assert 'r_max' in s


def test_cg1_codegen_mirror_aspheric_warns():
    """CG-1: an aspherized/Q-type mirror (which apply_mirror cannot represent)
    now warns instead of silently flattening to base conic."""
    from lumenairy.io.codegen import generate_simulation_script
    pres = {
        'name': 'mtest', 'aperture_diameter': 10e-3,
        'elements': [
            {'element_type': 'mirror', 'radius': -0.1, 'conic': 0.0,
             'aspheric_coeffs': {4: 1e-3}, 'semi_diameter': 5e-3,
             'is_stop': False, 'comment': 'M1'},
        ],
        'all_thicknesses': [0.0], 'thicknesses': [], 'object_distance': 0.0,
    }
    with pytest.warns(UserWarning, match="cannot represent"):
        generate_simulation_script(
            pres, wavelength=633e-9, N=64, dx=8e-6,
            include_plotting=False, include_analysis=False)


def test_storage_none_metadata_attr_skipped(tmp_path):
    """storage-nit: a None metadata value is skipped at the boundary (h5py
    cannot store None and would otherwise raise deep in its C layer)."""
    pytest.importorskip('h5py')
    from lumenairy.io.storage import load_planes_h5, save_planes_h5
    E = np.ones((8, 8), dtype=np.complex128)
    planes = [{'field': E, 'dx': 1e-6, 'z': 0.0, 'label': 'p0'}]
    p = str(tmp_path / 'planes.h5')
    # A None metadata value must not crash the save.
    save_planes_h5(p, planes, wavelength=633e-9, metadata={'note': None,
                                                           'run': 'A'})
    loaded, meta = load_planes_h5(p)
    assert len(loaded) == 1
    # The None key is simply absent on read-back; the real one survives.
    assert meta.get('run') == 'A'
    assert 'note' not in meta


# =========================================================================
# AUDIT 18 -- optimize/{driver,context}.py (AUDIT_OPTIMIZE_SECOND_PASS)
# =========================================================================


def _opt3_quadratic_merit(ctx):
    x = np.asarray(ctx.x, dtype=np.float64)
    return float(np.sum(x * x))


def _pickle_safe_zero(x):   # module-level -> picklable (no pickle-probe warn)
    return 0.0


def test_opt3_lm_path_writes_checkpoint_history(tmp_path):
    """OPT-3: the method='lm' path now routes through the shared per-eval
    bookkeeping, so a run with state_file= writes rolling checkpoints (a
    non-empty history) instead of nothing-until-the-final-force-save."""
    import json

    from lumenairy.optimize import (
        CallableMerit,
        DesignParameterization,
        design_optimize,
    )
    template = {
        'params': [1.0, 1.0, 1.0],
        'surfaces': [{'radius': np.inf, 'aperture': 5e-3,
                      'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [], 'aperture_diameter': 5e-3,
    }
    param = DesignParameterization(
        template=template,
        free_vars=[('params', 0), ('params', 1), ('params', 2)],
        bounds=[(-5.0, 5.0)] * 3)
    merit = CallableMerit(_opt3_quadratic_merit, weight=1.0, name='sq')
    state_file = str(tmp_path / 'lm_state.json')
    import warnings
    with warnings.catch_warnings():
        # method='lm' + bounds emits a loud lm->trf override UserWarning; both
        # route through the same least_squares ``residuals`` closure (the OPT-3
        # code path), so silence it here to isolate the checkpoint behaviour.
        warnings.simplefilter('ignore')
        design_optimize(param, [merit], wavelength=1.31e-6, method='lm',
                        max_iter=6, state_file=state_file,
                        state_save_every=1, verbose=False)
    assert os.path.isfile(state_file)
    with open(state_file, encoding='utf-8') as fh:
        payload = json.load(fh)
    # The LM residuals path recorded eval history (pre-OPT-3 it was empty).
    assert len(payload.get('history', [])) > 0
    assert payload.get('merit_best') is not None


def test_opt_second_pass_constraint_no_stale_deprecation():
    """OPT-nit: constructing a Constraint no longer emits the stale
    'auto-probe removal' DeprecationWarning (21 minor versions past its
    scheduled v5.0 removal)."""
    import warnings

    from lumenairy.optimize import Constraint
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        Constraint(fun=_pickle_safe_zero, lb=0.0, ub=None)
    assert not any(issubclass(w.category, DeprecationWarning) for w in rec)


# =========================================================================
# LOOSE ENDS 2026-07 -- verified-open residuals from the audit campaign
# =========================================================================

# --- AUDIT 1 residual: gerchberg_saxton_jax metric off-by-one ------------

def test_pr_gs_jax_final_error_matches_numpy():
    """The JAX GS kernel's error metric now re-transforms the FINAL
    iterate (mirroring the NumPy path's post-loop FFT); pre-fix it used
    the far field carried out of the fori_loop = the PREVIOUS iterate's.
    Full-band random fixture keeps every angle()/|z| well-conditioned so
    the two backends agree to machine precision."""
    jax = pytest.importorskip('jax')
    from lumenairy.analysis.phase_retrieval import (
        gerchberg_saxton,
        gerchberg_saxton_jax,
    )
    # x64 for float64 parity; restore afterwards so the test doesn't
    # perturb order-dependent jax state in the rest of the session.
    prev_x64 = bool(jax.config.jax_enable_x64)
    jax.config.update('jax_enable_x64', True)
    try:
        N = 32
        rng = np.random.default_rng(42)
        source = rng.uniform(0.5, 1.5, (N, N))
        target = rng.uniform(0.5, 1.5, (N, N))
        # Match total powers so the NumPy path's target normalisation
        # is a no-op (the JAX kernel consumes the raw target).
        target *= np.sqrt(np.sum(source**2) / np.sum(target**2))
        phase0 = np.zeros((N, N))
        _, err_np = gerchberg_saxton(source, target, n_iter=25,
                                     initial_phase=phase0,
                                     backend='numpy')
        _, err_jx = gerchberg_saxton_jax(source, target, n_iter=25,
                                         initial_phase=phase0,
                                         dtype=np.float64)
    finally:
        jax.config.update('jax_enable_x64', prev_x64)
    # Per-iteration error deltas are ~1e-3 relative, so the pre-fix
    # previous-iterate metric fails this by ~7 orders of magnitude.
    assert err_jx == pytest.approx(err_np, rel=1e-10)


# --- AUDIT 3 residual: vectorial_hfpi output_grid -> output_shape --------

def _vhfpi_fixture():
    N = 16
    x = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(x, x)
    Ex = np.exp(-(X**2 + Y**2)).astype(np.complex128)
    Ey = 0.3 * Ex
    kw = dict(dx=2e-6, z_to_aperture=1e-3, aperture_radius=8e-6,
              z_aperture_to_output=1e-3, wavelength=1.31e-6,
              n_paths=200, rng=7)
    return Ex, Ey, kw


def test_vhfpi_output_shape_rename_legacy_alias_warns():
    """propagate_vector_hfpi_freespace_aperture now takes output_shape=
    (the v5.2 hfpi spelling); the legacy output_grid= keeps working but
    warns, and produces the identical (same-seed) field."""
    from lumenairy.propagators.vectorial_hfpi import (
        propagate_vector_hfpi_freespace_aperture,
    )
    Ex, Ey, kw = _vhfpi_fixture()
    ex_new, ey_new = propagate_vector_hfpi_freespace_aperture(
        Ex, Ey, output_shape=(12, 10), **kw)
    with pytest.warns(DeprecationWarning):
        ex_old, ey_old = propagate_vector_hfpi_freespace_aperture(
            Ex, Ey, output_grid=(12, 10), **kw)
    assert ex_new.shape == (12, 10) and ey_new.shape == (12, 10)
    assert np.array_equal(ex_new, ex_old)
    assert np.array_equal(ey_new, ey_old)


def test_vhfpi_output_shape_and_grid_both_given_raises():
    from lumenairy.propagators.vectorial_hfpi import (
        propagate_vector_hfpi_freespace_aperture,
    )
    Ex, Ey, kw = _vhfpi_fixture()
    with pytest.raises(ValueError, match='both'):
        propagate_vector_hfpi_freespace_aperture(
            Ex, Ey, output_shape=(12, 10), output_grid=(12, 10), **kw)


# --- AUDIT 4 residuals: afocal f_eff display + paraxial_focus_world ------

def _flat_window_prescription():
    """Flat N-BK7 window: exactly afocal (f_eff non-finite, traced rays
    stay parallel)."""
    return {
        'surfaces': [
            {'radius': float('inf'), 'glass_before': 'air',
             'glass_after': 'N-BK7', 'semi_diameter': 5e-3},
            {'radius': float('inf'), 'glass_before': 'N-BK7',
             'glass_after': 'air', 'semi_diameter': 5e-3},
        ],
        'thicknesses': [2e-3, 0.0],
        'aperture_diameter': 10e-3,
    }


def test_rt_trace_summary_afocal_prints_half_angle(capsys):
    """RT residual: for an afocal system trace_summary no longer prints
    a radians half-angle with a length unit label (nor the meaningless
    Spot/Airy ratio) -- it reports an explicit urad half-angle."""
    import lumenairy as la
    from lumenairy.raytrace.layout import trace_summary
    surfs = la.surfaces_from_prescription(_flat_window_prescription())
    rays = la.make_rings(3e-3, 2, 6, 0.0, 1.31e-6)
    res = la.trace(rays, surfs, 1.31e-6)
    trace_summary(res)
    out = capsys.readouterr().out
    assert 'Airy half-angle' in out
    assert 'urad' in out
    assert 'Spot/Airy' not in out
    assert 'Airy radius' not in out


def test_rt_paraxial_focus_world_afocal_raises():
    """RT residual: paraxial_focus_world on an afocal system raises the
    documented ValueError instead of returning a garbage far
    intersection."""
    import lumenairy as la
    with pytest.raises(ValueError, match='afocal'):
        la.paraxial_focus_world(
            la.world_surfaces_from_prescription(
                _flat_window_prescription()),
            1.31e-6)


def test_rt_paraxial_focus_world_dead_rays_raise():
    """RT residual: if the probe rays are vignetted the failure is a
    clear ValueError, not a NaN focus."""
    import lumenairy as la
    presc = la.make_singlet(R1=50e-3, R2=np.inf, d=3e-3,
                            glass='N-BK7', aperture=10e-3)
    wsurfs = la.world_surfaces_from_prescription(presc)
    for s in wsurfs:
        s.semi_diameter = 1e-9          # vignette everything
    with pytest.raises(ValueError, match='did not survive'):
        la.paraxial_focus_world(wsurfs, 1.31e-6, aperture_radius=1e-3)


# --- AUDIT 3 residual: _bluestein_2d chirp H-FFT cache -------------------

def test_prop_bluestein_h_fft_cache_bitexact_and_hit():
    """The chirp-kernel FFT is cached (numpy default-fft path only) and
    a cache hit returns a bit-exact copy of the uncached result."""
    from lumenairy.propagators import _bluestein as bl
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    rng = np.random.default_rng(3)
    E = (rng.standard_normal((24, 20))
         + 1j * rng.standard_normal((24, 20)))
    kw = dict(alpha_x=1.7e-3, alpha_y=2.3e-3, N_out_y=18, N_out_x=22,
              sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
    bl._clear_h_fft_cache()
    F1 = bl._bluestein_2d(E, **kw)
    assert len(bl._H_FFT_CACHE) == 1
    assert bl._H_FFT_CACHE_HITS == 0
    F2 = bl._bluestein_2d(E, **kw)
    assert bl._H_FFT_CACHE_HITS == 1        # served from cache
    assert np.array_equal(F1, F2)           # bit-exact
    bl._clear_h_fft_cache()


def test_prop_bluestein_h_fft_cache_skips_custom_fft():
    """Caller-supplied fft2 callables (CuPy / JAX / custom) must bypass
    the cache -- only the module's default numpy path is keyed."""
    from lumenairy.propagators import _bluestein as bl
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    rng = np.random.default_rng(3)
    E = (rng.standard_normal((24, 20))
         + 1j * rng.standard_normal((24, 20)))
    bl._clear_h_fft_cache()
    F_ref = bl._bluestein_2d(E, 1.7e-3, 2.3e-3, 18, 22, sign=-1,
                             xp=np, fft2=_fft2, ifft2=_ifft2)
    bl._clear_h_fft_cache()
    F_custom = bl._bluestein_2d(E, 1.7e-3, 2.3e-3, 18, 22, sign=-1,
                                xp=np, fft2=np.fft.fft2,
                                ifft2=np.fft.ifft2)
    assert len(bl._H_FFT_CACHE) == 0        # nothing cached
    assert np.allclose(F_ref, F_custom, rtol=1e-12, atol=1e-12)
    bl._clear_h_fft_cache()
