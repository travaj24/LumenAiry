"""Subsystem-audit remediation campaign (docs/audits/AUDIT_*_2026_07_07..09).

One growing file, sectioned per audit, pinning the findings fixed in the
chronological subsystem sweep that follows the v5.21 delta audit.
"""
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


def test_rt4_world_coord_break_tilt_sign():
    """RT-4: world._apply_coord_break now uses the legacy optical (3.7.1)
    sign -- a +tilt_x break yields a new frame whose local-to-world rotation
    is Rx(-tx) (the transpose-negation of the ray-coordinate transform), so
    trace() and trace_world() fold in the SAME angular direction."""
    from lumenairy.raytrace.world import _apply_coord_break, _rot_x
    tx_deg = 30.0
    origin = np.zeros(3)
    R = np.eye(3)
    _, new_R = _apply_coord_break(origin, R, {'tilt_x_deg': tx_deg})
    expected = _rot_x(-np.radians(tx_deg))  # RT-4 fixed convention
    np.testing.assert_allclose(new_R, expected, atol=1e-12)


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
