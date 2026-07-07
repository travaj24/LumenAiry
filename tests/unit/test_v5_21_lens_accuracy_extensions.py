"""v5.21 lens-propagator accuracy extensions -- correctness gates.

Accuracy features added to the real-lens propagators (Maslov / GBD / traced /
analytic).  Each gate proves the feature is a real accuracy improvement that
does NOT change the converged physics (validated against a trusted reference).
"""
import numpy as np
import pytest

LAM = 0.633e-6


def _relerr(A, B):
    return float(np.linalg.norm(A - B) / (np.linalg.norm(B) + 1e-300))


def _gauss(N, dx, w0=1.8e-3):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _singlet(last_gap):
    """Plano-convex N-BK7 singlet; ``last_gap`` = air gap after the lens."""
    return {'aperture_diameter': 6e-3, 'surfaces': [
        {'radius': 25e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [3e-3, last_gap, 0.0]}


# ==========================================================================
# Maslov poly_order='auto' -- adaptive tensor-Chebyshev fit order
# ==========================================================================
def test_maslov_auto_selector_recovers_known_degree():
    """The order selector recovers the true total-degree of a synthetic OPD
    that is EXACTLY a tensor-Chebyshev polynomial of that degree, and picks the
    floor order for a smooth (low-degree) OPD -- i.e. it neither under- nor
    over-fits."""
    from lumenairy._math.chebyshev import chebyshev_vandermonde
    from lumenairy.elements.lenses import _multi_indices_total_degree
    from lumenairy.elements.lenses_maslov import (
        _MZ_POLY_AUTO_MAX,
        _MZ_POLY_AUTO_MIN,
        _MZ_POLY_AUTO_RTOL,
        _MZ_POLY_AUTO_TARGET,
        _select_poly_order_auto,
    )

    rng = np.random.default_rng(0)
    n = 4000
    u1, u2, u3, u4 = (rng.uniform(-1, 1, n) for _ in range(4))

    def synth(true_order):
        mi = _multi_indices_total_degree(4, true_order)
        T = [chebyshev_vandermonde(u, true_order) for u in (u1, u2, u3, u4)]
        c = rng.standard_normal(len(mi))
        opd = np.zeros_like(u1)
        for cj, (k1, k2, k3, k4) in zip(c, mi):
            w = 0.3 ** (k1 + k2 + k3 + k4)   # low-order dominated
            opd += cj * w * T[0][k1] * T[1][k2] * T[2][k3] * T[3][k4]
        return opd

    for true_order in (3, 4, 5, 6):
        p, res = _select_poly_order_auto(
            u1, u2, u3, u4, synth(true_order),
            order_min=_MZ_POLY_AUTO_MIN, order_max=_MZ_POLY_AUTO_MAX,
            target_waves=1e-12, rtol=_MZ_POLY_AUTO_RTOL)
        # a genuine degree-`true_order` poly must not be under-fit
        assert p >= min(true_order, _MZ_POLY_AUTO_MAX), (true_order, p)
        assert res < 1e-9   # exact poly -> residual at fit-noise floor

    # smooth (degree-2) OPD -> the cheap floor order
    p, _ = _select_poly_order_auto(
        u1, u2, u3, u4, synth(2),
        order_min=_MZ_POLY_AUTO_MIN, order_max=_MZ_POLY_AUTO_MAX,
        target_waves=_MZ_POLY_AUTO_TARGET, rtol=_MZ_POLY_AUTO_RTOL)
    assert p == _MZ_POLY_AUTO_MIN


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_maslov_auto_matches_high_order_reference():
    """poly_order='auto' converges the field to the high-order reference (it
    raises the order until the OPD fit stops improving), and is markedly better
    than a fixed low order -- so 'auto' is a genuine accuracy win, not a
    physics change."""
    from lumenairy.elements.lenses_maslov import (
        _MZ_POLY_AUTO_MAX,
        apply_real_lens_maslov,
    )
    N, dx = 64, 5e-6
    E = _gauss(N, dx)
    kw = dict(prescription=_singlet(2e-3), wavelength=LAM, dx=dx,
              integration_method='quadrature', n_v2=48)
    F_auto = apply_real_lens_maslov(E, poly_order='auto', **kw)
    F_ref = apply_real_lens_maslov(E, poly_order=_MZ_POLY_AUTO_MAX, **kw)
    F_lo = apply_real_lens_maslov(E, poly_order=4, **kw)
    # auto tracks the converged high-order field...
    assert _relerr(F_auto, F_ref) < 1e-3
    # ...and is a real improvement over the fixed low order it started from
    assert _relerr(F_auto, F_ref) < 0.5 * _relerr(F_lo, F_ref)


def test_maslov_auto_rejects_bad_poly_order():
    """A non-'auto' string poly_order fails fast with a clear message."""
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    N, dx = 32, 5e-6
    E = _gauss(N, dx)
    with pytest.raises(ValueError, match="poly_order must be an int or 'auto'"):
        apply_real_lens_maslov(E, prescription=_singlet(2e-3), wavelength=LAM,
                               dx=dx, poly_order='quadratic')


# ==========================================================================
# GBD converge_gbd_sampling -- beamlet-width convergence vs exact ASM
# ==========================================================================
def _gauss_gbd(N, dx, w0):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def test_converge_gbd_sampling_is_exported_and_wellformed():
    """The helper is exported at the top level and returns the documented
    fields, with the reported best overlap actually the minimum of the swept
    error curve, and the achieved error validated against the EXACT ASM."""
    from lumenairy import converge_gbd_sampling
    N, dx = 128, 4e-6
    E = _gauss_gbd(N, dx, 0.12e-3)
    r = converge_gbd_sampling(E, dx, wavelength=LAM, test_distance=8e-3)
    for key in ('overlap', 'waist_factor', 'sample_step', 'error', 'errors',
                'converged', 'n_beamlets', 'reference'):
        assert key in r
    assert r['reference'] == 'asm'
    # the returned best overlap is the argmin of the reported error curve
    assert r['errors'][r['overlap']] == min(r['errors'].values())
    assert r['error'] == r['errors'][r['overlap']]
    # w0 = overlap * sample_step (the physically meaningful width)
    assert abs(r['waist_factor'] - r['overlap'] * r['sample_step']) < 1e-9
    # a smooth Gaussian free-space leg is well-represented by GBD (vs exact ASM)
    assert r['error'] < 0.05


def test_converge_gbd_sampling_finds_real_improvement():
    """Sweeping the width finds a beamlet setting strictly better than a poorly
    chosen (over-wide) one -- i.e. the convergence is a real accuracy lever, not
    a no-op."""
    from lumenairy.propagators.gbd import converge_gbd_sampling
    N, dx = 128, 4e-6
    E = _gauss_gbd(N, dx, 0.05e-3)
    r = converge_gbd_sampling(E, dx, wavelength=LAM, test_distance=6e-3,
                              overlaps=(2.5, 2.0, 1.5, 1.2, 1.0, 0.8))
    assert r['error'] < r['errors'][2.5]      # best beats the over-wide beamlet
    assert r['error'] < 0.03


def test_converge_gbd_sampling_supplied_reference():
    """A caller-supplied reference (e.g. a higher-fidelity solver) is used
    instead of the ASM oracle, and is reported as such."""
    from lumenairy.propagators.asm import angular_spectrum_propagate
    from lumenairy.propagators.gbd import converge_gbd_sampling
    N, dx = 96, 4e-6
    E = _gauss_gbd(N, dx, 0.12e-3)
    ref = angular_spectrum_propagate(E, 5e-3, LAM, dx)
    r = converge_gbd_sampling(E, dx, wavelength=LAM, test_distance=5e-3,
                              reference=ref, overlaps=(1.5, 1.0))
    assert r['reference'] == 'supplied'
    assert r['error'] < 0.05


# ==========================================================================
# GBD longitudinal E_z vector beamlets (E . k = 0)
# ==========================================================================
def test_ez_primitive_exact_vs_transversality():
    """reconstruct_vector_field_with_ez reproduces the exact transversality
    relation Ez = -(L*Ex + M*Ey)/N to machine precision.  For a beam tilted by
    theta in x with Ey = 0 this is Ez = -tan(theta)*Ex, verified up to NA 0.5."""
    import dataclasses

    from lumenairy.propagators.gbd import (
        decompose_field_to_beamlets,
        reconstruct_vector_field_with_ez,
    )
    N, dx = 160, 1e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    A = np.exp(-(X ** 2 + Y ** 2) / (25e-6) ** 2).astype(np.complex128)
    for sth in (0.1, 0.3, 0.5):
        cth = np.sqrt(1 - sth ** 2)
        tan = sth / cth
        bx = decompose_field_to_beamlets(A, dx, wavelength=LAM, sample_step=1,
                                         waist_factor=1.5)
        dirs = np.zeros_like(bx.directions)
        dirs[:, 0] = sth
        dirs[:, 2] = cth
        bx = dataclasses.replace(bx, directions=dirs)
        by = dataclasses.replace(bx, amplitude=0.3 * bx.amplitude)  # Ey=0.3*Ex
        Ex, Ey, Ez = reconstruct_vector_field_with_ez(
            bx, by, Ny=N, Nx=N, dx=dx, wavelength=LAM, window=6.0)
        # M=0 so Ez = -(sinT/cosT)*Ex = -tan(T)*Ex exactly
        assert _relerr(Ez, -tan * Ex) < 1e-12
        m = np.abs(Ex) > 1e-3
        assert abs(float(np.median(np.abs(Ez)[m] / np.abs(Ex)[m])) - tan) < 1e-6


def test_ez_grows_with_NA_and_wrapper_shape():
    """The longitudinal energy fraction grows monotonically with tilt/NA (the
    physics a transverse-only GBD misses), and the free-space wrapper returns a
    (3, Ny, Nx) field with a negligible Ez for a paraxial (untilted) source."""
    import dataclasses

    from lumenairy.propagators.gbd import (
        decompose_field_to_beamlets,
        propagate_gbd_freespace_vector,
        reconstruct_vector_field_with_ez,
    )
    N, dx = 160, 1e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    A = np.exp(-(X ** 2 + Y ** 2) / (25e-6) ** 2).astype(np.complex128)
    fracs = []
    for sth in (0.1, 0.3, 0.5):
        cth = np.sqrt(1 - sth ** 2)
        bx = decompose_field_to_beamlets(A, dx, wavelength=LAM, sample_step=1,
                                         waist_factor=1.5)
        dirs = np.zeros_like(bx.directions)
        dirs[:, 0] = sth
        dirs[:, 2] = cth
        bx = dataclasses.replace(bx, directions=dirs)
        by = dataclasses.replace(bx, amplitude=np.zeros_like(bx.amplitude))
        Ex, Ey, Ez = reconstruct_vector_field_with_ez(
            bx, by, Ny=N, Nx=N, dx=dx, wavelength=LAM, window=6.0)
        long_frac = float(np.sum(np.abs(Ez) ** 2)
                          / np.sum(np.abs(Ex) ** 2 + np.abs(Ez) ** 2))
        fracs.append(long_frac)
    assert fracs[0] < fracs[1] < fracs[2]      # more NA -> more longitudinal

    # free-space wrapper: (3, N, N), and a paraxial untilted source has ~no Ez
    Evec = np.stack([A, np.zeros_like(A)], axis=0)
    out = propagate_gbd_freespace_vector(
        Evec, dx, z=0.0, wavelength=LAM, return_longitudinal=True,
        direction_sampling=True)
    assert out.shape == (3, N, N)
    assert np.abs(out[2]).max() / np.abs(out[0]).max() < 1e-6


# ==========================================================================
# traced blind angular segmentation (multi-congruence combined field)
# ==========================================================================
def _two_beam_scene(N, dx):
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)

    def beam(sth, off):
        return (np.exp(-((X - off) ** 2 + Y ** 2) / (22e-6) ** 2)
                * np.exp(1j * 2 * np.pi / LAM * sth * X)).astype(np.complex128)

    E1 = beam(0.05, -18e-6)
    E2 = beam(-0.04, 18e-6)
    return E1, E2


def _aberrated_singlet():
    return {'aperture_diameter': 7e-3, 'surfaces': [
        {'radius': 12e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': -60e-3, 'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [4e-3, 10e-3]}


def test_traced_segmentation_partition_sums_to_input():
    """The angular partition is exact: with min_segment_power=0 the segment
    fields sum back to the input field to machine precision, and two beams
    separated in angle are auto-split into exactly two segments (a single beam
    into one)."""
    from lumenairy.elements._lens_traced import apply_real_lens_traced_segmented
    N, dx = 96, 2e-6
    E1, E2 = _two_beam_scene(N, dx)
    E = E1 + E2
    kw = dict(prescription=_aberrated_singlet(), wavelength=LAM, dx=dx)
    segs = apply_real_lens_traced_segmented(
        E, return_segments=True, min_segment_power=0.0, **kw)
    assert len(segs) == 2                       # auto-detected the angular gap
    S = np.zeros_like(E)
    for s in segs:
        S = S + s
    assert _relerr(S, E) < 1e-12
    # a single beam is unimodal -> one segment (== plain traced, no fragmenting)
    segs1 = apply_real_lens_traced_segmented(E1, return_segments=True, **kw)
    assert len(segs1) == 1


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_traced_segmentation_matches_known_emitter_reference():
    """For a genuinely multi-congruence field (two crossing beams) the blind
    gap-split traced matches the exact per-emitter reference
    (apply_real_lens_traced_multi with the KNOWN separated beams), and is orders
    of magnitude better than the single-congruence-violating naive
    traced(sum)."""
    from lumenairy.elements._lens_traced import (
        apply_real_lens_traced,
        apply_real_lens_traced_multi,
        apply_real_lens_traced_segmented,
    )
    N, dx = 96, 2e-6
    E1, E2 = _two_beam_scene(N, dx)
    E = E1 + E2
    kw = dict(prescription=_aberrated_singlet(), wavelength=LAM, dx=dx)
    ref = apply_real_lens_traced_multi([E1, E2], carriers='auto', **kw)
    naive = apply_real_lens_traced(E, carrier='auto', **kw)
    blind = apply_real_lens_traced_segmented(E, **kw)
    # naive violates the single-congruence assumption -> large error
    assert _relerr(naive, ref) > 0.5
    # blind gap-split recovers the per-congruence answer
    assert _relerr(blind, ref) < 5e-3
    assert _relerr(blind, ref) < 0.01 * _relerr(naive, ref)


# ==========================================================================
# traced jax twin: differentiable w.r.t. prescription geometry (lens design)
# ==========================================================================
def _jax_ok():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_traced_jax_differentiable_wrt_prescription_geometry():
    """apply_real_lens_traced_jax is differentiable w.r.t. the prescription
    radius / thickness (the lens-design gradient on the ray-traced OPD): a
    phase-sensitive loss's jax.grad matches finite-difference.  The static path
    (no radii/conics/thicknesses) is unchanged and finite."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.elements._lens_jax import apply_real_lens_traced_jax
    N, dx = 48, 4e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E0 = jnp.asarray(np.exp(-(X ** 2 + Y ** 2) / (60e-6) ** 2)
                     .astype(np.complex128))
    presc = {'aperture_diameter': 4e-3, 'surfaces': [
        {'radius': 20e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [3e-3, 8e-3]}

    base_r = jnp.array([20e-3, jnp.inf, jnp.inf])

    def loss_r(r):
        radii = base_r.at[0].set(r)
        E = apply_real_lens_traced_jax(E0, prescription=presc, wavelength=LAM,
                                       dx=dx, radii=radii)
        return jnp.abs(jnp.sum(E)) ** 2          # phase-sensitive

    g = float(jax.grad(loss_r)(20e-3))
    h = 2e-6
    fd = float((loss_r(20e-3 + h) - loss_r(20e-3 - h)) / (2 * h))
    assert abs(g - fd) / (abs(fd) + 1e-30) < 1e-5
    assert abs(g) > 1.0                          # a real, non-zero gradient

    base_t = jnp.array([3e-3, 8e-3])

    def loss_t(t):
        thk = base_t.at[1].set(t)
        E = apply_real_lens_traced_jax(E0, prescription=presc, wavelength=LAM,
                                       dx=dx, thicknesses=thk)
        return jnp.abs(jnp.sum(E)) ** 2

    gt = float(jax.grad(loss_t)(8e-3))
    fdt = float((loss_t(8e-3 + h) - loss_t(8e-3 - h)) / (2 * h))
    assert abs(gt - fdt) / (abs(fdt) + 1e-30) < 1e-5

    # static path (no differentiable geometry) still produces a finite field
    Es = apply_real_lens_traced_jax(E0, prescription=presc, wavelength=LAM,
                                    dx=dx)
    assert np.isfinite(float(jnp.sum(jnp.abs(Es) ** 2)))


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_traced_jax_diff_geometry_requires_input_amplitude():
    """Differentiable geometry requires amplitude='input' (the analytic-
    amplitude callback carries no prescription gradient) and says so."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.elements._lens_jax import apply_real_lens_traced_jax
    N, dx = 32, 4e-6
    E0 = jnp.asarray(np.ones((N, N), dtype=np.complex128))
    presc = {'aperture_diameter': 4e-3, 'surfaces': [
        {'radius': 20e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [3e-3, 8e-3]}
    with pytest.raises(ValueError, match="amplitude='input'"):
        apply_real_lens_traced_jax(
            E0, prescription=presc, wavelength=LAM, dx=dx,
            radii=jnp.array([20e-3, jnp.inf, jnp.inf]), amplitude='analytic')


# ==========================================================================
# GBD complex-source-point (CSP) non-paraxial beamlets
# ==========================================================================
def _asm_np(E, z, wavelength, dx):
    N = E.shape[0]
    fx = np.fft.fftfreq(N, dx)
    FX, FY = np.meshgrid(fx, fx)
    k = 2 * np.pi / wavelength
    kz = np.sqrt((k * k - (2 * np.pi * FX) ** 2
                  - (2 * np.pi * FY) ** 2).astype(complex))
    return np.fft.ifft2(np.fft.fft2(E) * np.exp(1j * kz * z))


def test_csp_field_is_exact_helmholtz_vs_asm_at_high_NA():
    """A single complex-source-point beam is an EXACT scalar-Helmholtz field:
    its waist propagated by the angular-spectrum method matches the analytic CSP
    field to grid precision at ALL NA, while a paraxial Gaussian degrades badly
    (this is the whole point of the non-paraxial beamlet)."""
    from lumenairy.propagators.gbd import csp_beamlet_field
    lam = 0.633e-6

    def csp(X, Y, Zc, w0):
        return csp_beamlet_field(
            X.ravel(), Y.ravel(), Zc,
            np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]),
            np.array([w0]), lam, np.array([1.0 + 0j])).reshape(X.shape)

    for w0 in (1.0e-6, 0.45e-6):                 # NA ~ 0.20 and ~0.45
        N = 512
        dx = min(w0 / 4.0, lam / 3.0)
        xs = (np.arange(N) - N // 2) * dx
        X, Y = np.meshgrid(xs, xs)
        z = 6.0 * (np.pi * w0 ** 2 / lam)
        E0 = csp(X, Y, 0.0, w0)
        Ez_exact = csp(X, Y, z, w0)
        Ez_asm = _asm_np(E0, z, lam, dx)
        m = (np.abs(X) < 0.35 * N * dx) & (np.abs(Y) < 0.35 * N * dx)
        # CSP is exact-Helmholtz -> matches ASM to grid precision
        assert _relerr(Ez_asm[m], Ez_exact[m]) < 2e-3
    # at the highest NA the paraxial Gaussian is an order(s)-of-magnitude worse
    w0 = 0.45e-6
    N, dx = 512, min(w0 / 4.0, lam / 3.0)
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    z = 6.0 * (np.pi * w0 ** 2 / lam)
    zR = np.pi * w0 ** 2 / lam
    q = z - 1j * zR
    par = (-1j * zR) / q * np.exp(1j * 2 * np.pi / lam * z
                                  + 1j * 2 * np.pi / lam * (X ** 2 + Y ** 2)
                                  / (2 * q))
    m = (np.abs(X) < 0.35 * N * dx) & (np.abs(Y) < 0.35 * N * dx)
    csp_err = _relerr(_asm_np(csp(X, Y, 0.0, w0), z, lam, dx)[m],
                      csp(X, Y, z, w0)[m])
    par_err = _relerr(par[m], _asm_np(csp(X, Y, 0.0, w0), z, lam, dx)[m])
    assert par_err > 50 * csp_err                # paraxial fails, CSP does not


def test_csp_gbd_propagator_runs_and_is_competitive():
    """The CSP-GBD free-space propagator runs and reconstructs a propagated
    field at least as accurately as the paraxial GBD sum (per-beamlet exactness
    never makes the sum worse; the reconstruction floor is shared)."""
    from lumenairy.propagators.asm import angular_spectrum_propagate
    from lumenairy.propagators.gbd import (
        match_global_phase,
        propagate_gbd_freespace,
        propagate_gbd_freespace_csp,
    )
    N, dx, w0s = 96, 1.0e-6, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E0 = np.exp(-(X ** 2 + Y ** 2) / w0s ** 2).astype(np.complex128)
    z = 4.0 * np.pi * w0s ** 2 / LAM
    ref = angular_spectrum_propagate(E0, z, LAM, dx)
    m = (np.abs(X) < 0.35 * N * dx) & (np.abs(Y) < 0.35 * N * dx)
    E_csp = match_global_phase(
        propagate_gbd_freespace_csp(E0, dx, z=z, wavelength=LAM,
                                    waist_factor=1.5, sample_step=1), ref)
    E_par = match_global_phase(
        propagate_gbd_freespace(E0, dx, z=z, wavelength=LAM, waist_factor=1.5,
                                sample_step=1, direction_sampling=True), ref)
    assert E_csp.shape == (N, N)
    assert _relerr(E_csp[m], ref[m]) < 0.06
    # CSP never worse than paraxial (shared reconstruction floor)
    assert _relerr(E_csp[m], ref[m]) <= _relerr(E_par[m], ref[m]) + 5e-3


def test_csp_windowed_matches_dense():
    """The windowed CSP reconstruction (evaluate each beamlet only over its
    local box) matches the dense O(n*N^2) sum to the tail truncation."""
    from lumenairy.propagators.gbd import propagate_gbd_freespace_csp
    N, dx, w0s = 96, 1.0e-6, 6e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E0 = np.exp(-(X ** 2 + Y ** 2) / w0s ** 2).astype(np.complex128)
    z = 4.0 * np.pi * w0s ** 2 / LAM
    dense = propagate_gbd_freespace_csp(E0, dx, z=z, wavelength=LAM,
                                        sample_step=2)
    win = propagate_gbd_freespace_csp(E0, dx, z=z, wavelength=LAM,
                                      sample_step=2, window=6.0)
    assert _relerr(win, dense) < 1e-9


# ==========================================================================
# Turnkey lens-design optimizer over the differentiable traced jax twin
# ==========================================================================
@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_optimize_traced_geometry_reduces_loss():
    """The turnkey optimizer drives a merit down through the differentiable
    prescription geometry (grad flows trace->fit->Newton->OPD->merit->Adam),
    and returns an updated prescription."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.optimize import optimize_traced_geometry
    N, dx = 32, 4e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E_in = np.exp(-(X ** 2 + Y ** 2) / (60e-6) ** 2).astype(np.complex128)
    presc = {'aperture_diameter': 4e-3, 'surfaces': [
        {'radius': 20e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [3e-3, 8e-3]}

    def merit(E):
        return -jnp.abs(jnp.sum(E)) ** 2 / (N * N)

    res = optimize_traced_geometry(
        E_in, presc, wavelength=LAM, dx=dx, optimize=('radii',),
        merit=merit, n_steps=6)
    assert res['loss_history'][-1] < res['loss_history'][0]     # improved
    # the free (finite) radius actually moved; the flat surface stayed inf
    assert res['prescription']['surfaces'][0]['radius'] != 20e-3
    assert not np.isfinite(res['prescription']['surfaces'][1]['radius'])


# ==========================================================================
# Uniform fold-Airy (CFU) caustic-finite oscillatory-integral evaluator
# ==========================================================================
def test_uniform_fold_airy_matches_exact_cubic_and_stays_finite():
    """The uniform fold evaluator reproduces the EXACT cubic-phase integral
    int (1+a t) exp(ik(t^3/3 - c t)) dt = 2pi k^-1/3 Ai(-k^2/3 c)
                                          - a 2pi i k^-2/3 Ai'(-k^2/3 c)
    to machine precision for a1=0 AND a1!=0, and stays finite through the
    caustic c->0 (where ordinary stationary phase diverges)."""
    from scipy.special import airy

    from lumenairy.elements.lenses_maslov import uniform_fold_airy

    def exact(k, c, a=0.0):
        ai, aip, _, _ = airy(-(k ** (2.0 / 3.0)) * c)
        return (2 * np.pi * k ** (-1.0 / 3.0) * ai
                - a * 2j * np.pi * k ** (-2.0 / 3.0) * aip)

    def f(t, c):
        return t ** 3 / 3 - c * t

    def fpp(t):
        return 2 * t

    worst = 0.0
    for k in (2000.0, 8000.0):
        for c in (0.03, 0.06):
            for a in (0.0, 0.5, -0.8):     # a != 0 exercises the a1/Ai' term
                s = np.sqrt(c)
                uni = uniform_fold_airy(k, -s, s, f(-s, c), f(s, c),
                                        fpp(-s), fpp(s), 1 + a * (-s), 1 + a * s)
                worst = max(worst, _relerr(uni, exact(k, c, a)))
    assert worst < 1e-10

    # through the caustic: finite and matches exact Airy down to c ~ 0
    k = 5000.0
    for c in (0.02, 0.005, 1e-6):
        s = np.sqrt(c)
        uni = uniform_fold_airy(k, -s, s, f(-s, c), f(s, c), fpp(-s), fpp(s))
        assert _relerr(uni, exact(k, c)) < 1e-9
        assert np.isfinite(uni)


def _mini_fast_singlet():
    """f/2.4 strong-SA plano-convex mini-singlet (R=2mm N-BK7, BFL~3.2mm)."""
    return {'aperture_diameter': 1.6e-3, 'surfaces': [
        {'radius': 2.0e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7', 'glass_after': 'air'},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [1.0e-3, 0.0]}


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_traced_multibranch_energy_and_gouy_through_focus():
    """The multi-branch traced field conserves energy in the single-branch
    regions and carries the correct KMAH (Gouy) index through focus: m=0
    before any caustic crossing, m=2 (phase -pi) for rays past both astigmatic
    focal-line crossings of the exit leg -- the discrete Gouy anomaly."""
    from lumenairy.elements._lens_traced_multibranch import (
        apply_real_lens_traced_multibranch,
    )
    N, dx = 192, 10e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E_in = np.exp(-(X ** 2 + Y ** 2) / (0.6e-3) ** 2).astype(np.complex128)
    P_in = float(np.sum(np.abs(E_in) ** 2))
    presc = _mini_fast_singlet()
    # pre-focus: single branch everywhere, m=0, energy ~conserved (2.7% is
    # the Gaussian tail beyond the launch rim)
    E1, d1 = apply_real_lens_traced_multibranch(
        E_in, prescription=presc, wavelength=LAM, dx=dx,
        output_plane_distance=2.4e-3, ray_subsample=1,
        return_diagnostics=True)
    assert np.all(np.isfinite(E1))
    assert abs(float(np.sum(np.abs(E1) ** 2)) / P_in - 1.0) < 0.08
    assert d1['n_branch'].max() == 1
    assert set(d1['kmah'].ravel().tolist()) == {0}
    # post-focus: all rays crossed both focal lines -> m=2 (Gouy -pi)
    E2, d2 = apply_real_lens_traced_multibranch(
        E_in, prescription=presc, wavelength=LAM, dx=dx,
        output_plane_distance=3.6e-3, ray_subsample=1,
        return_diagnostics=True)
    assert abs(float(np.sum(np.abs(E2) ** 2)) / P_in - 1.0) < 0.08
    assert 2 in set(d2['kmah'].ravel().tolist())
    assert 1 not in set(d2['kmah'].ravel().tolist())   # no half-crossed rays


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_traced_multibranch_matches_exact_diffraction_oracle():
    """Mid-annulus intensity profile matches the EXACT decouple-pipeline
    oracle (ray-traced exit-pupil field + direct Rayleigh-Sommerfeld sum --
    pointwise, aliasing-free) to <10% (masked off the axial caustic-band
    pixels, where ART is undefined by construction)."""
    from lumenairy.elements._lens_traced_multibranch import (
        _trace_launch_grid,
        apply_real_lens_traced_multibranch,
    )
    presc = _mini_fast_singlet()
    N, dx = 192, 10e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E_in = np.exp(-(X ** 2 + Y ** 2) / (0.6e-3) ** 2).astype(np.complex128)
    k0 = 2 * np.pi / LAM
    # exit-pupil field on a Nyquist-adequate launch grid (phase step < pi at
    # the output angles involved), then direct RS to the centre row
    g = _trace_launch_grid(presc, LAM, 0.5 * 1.6e-3 * 0.98, 1601, 0.0, 1.0)
    h = g['xs_in'][1] - g['xs_in'][0]
    ok = g['alive'] & np.isfinite(g['x_out']) & np.isfinite(g['opl'])
    d11, d12 = np.gradient(g['x_out'], h, h)
    d21, d22 = np.gradient(g['y_out'], h, h)
    Jex = d11 * d22 - d12 * d21
    ok &= np.isfinite(Jex) & (Jex > 0)
    Ein = np.exp(-(g['Xi'] ** 2 + g['Yi'] ** 2) / (0.6e-3) ** 2)
    wamp = (Ein * np.sqrt(np.abs(Jex))
            * np.exp(1j * k0 * g['opl']))[ok] * h * h
    px = g['x_out'][ok]
    py = g['y_out'][ok]
    d = 2.6e-3
    row = np.zeros(N, dtype=np.complex128)
    for i, xo in enumerate(xs):
        r = np.sqrt((xo - px) ** 2 + py ** 2 + d * d)
        row[i] = np.sum(wamp * np.exp(1j * k0 * r) / r * (d / r))
    ref = np.abs(row) ** 2
    mb = apply_real_lens_traced_multibranch(
        E_in, prescription=presc, wavelength=LAM, dx=dx,
        output_plane_distance=d, ray_subsample=1)
    c = N // 2
    a = ref / ref.max()
    b = np.abs(mb[c]) ** 2
    b = b / b.max()
    assert abs(int(np.argmax(ref)) - int(np.argmax(np.abs(mb[c])))) <= 1
    msk = a > 0.02
    msk[c - 3:c + 4] = False        # axial caustic band: ART undefined
    rel = float(np.linalg.norm(b[msk] - a[msk]) / np.linalg.norm(a[msk]))
    assert rel < 0.10


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_multibranch_ludwig_band_swap():
    """caustic_band='ludwig' swaps ONLY coalescing in-band pairs (k|dS|<=pi)
    for the uniform fold field: identical to 'plain' wherever there is no
    multipath, finite everywhere, and it TAMES the plain sum's fold
    over-amplification (peak does not increase)."""
    from lumenairy.elements._lens_traced_multibranch import (
        apply_real_lens_traced_multibranch,
    )
    presc = _mini_fast_singlet()
    N, dx = 192, 10e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E_in = np.exp(-(X ** 2 + Y ** 2) / (0.6e-3) ** 2).astype(np.complex128)
    kw = dict(prescription=presc, wavelength=LAM, dx=dx, ray_subsample=1)
    # single-branch plane: byte-identical
    Ep = apply_real_lens_traced_multibranch(
        E_in, output_plane_distance=2.4e-3, caustic_band='plain', **kw)
    El = apply_real_lens_traced_multibranch(
        E_in, output_plane_distance=2.4e-3, caustic_band='ludwig', **kw)
    assert np.array_equal(Ep, El)
    # multipath plane: swaps confined to multi-branch pixels, finite, tamed
    Ep, dg = apply_real_lens_traced_multibranch(
        E_in, output_plane_distance=2.9e-3, caustic_band='plain',
        return_diagnostics=True, **kw)
    El = apply_real_lens_traced_multibranch(
        E_in, output_plane_distance=2.9e-3, caustic_band='ludwig', **kw)
    changed = np.abs(El - Ep) > 1e-12
    assert changed.any()                          # the band exists here
    assert not changed[dg['n_branch'] < 2].any()  # only multi-branch pixels
    assert np.all(np.isfinite(El))
    assert np.abs(El).max() <= np.abs(Ep).max() + 1e-9


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_multibranch_input_carrier_tilt():
    """input_carrier: (0,0) is byte-identical to the default; 'auto' recovers
    a known carrier to sub-0.1%; a tilted input displaces the near-focus
    pattern by exactly the traced chief-ray transverse position (the carrier
    phase rides the branch eikonals, the envelope is sampled
    carrier-stripped); energy matches the untilted run."""
    import lumenairy.raytrace as rt
    from lumenairy.elements._lens_traced_multibranch import (
        apply_real_lens_traced_multibranch,
    )
    presc = _mini_fast_singlet()
    N, dx = 192, 10e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    env = np.exp(-(X ** 2 + Y ** 2) / (0.6e-3) ** 2)
    d = 3.6e-3   # post-focus: compact pattern, no exact-caustic plane
    kw = dict(prescription=presc, wavelength=LAM, dx=dx,
              output_plane_distance=d, ray_subsample=1)
    F0 = apply_real_lens_traced_multibranch(env.astype(complex), **kw)
    F00 = apply_real_lens_traced_multibranch(
        env.astype(complex), input_carrier=(0.0, 0.0), **kw)
    assert np.array_equal(F0, F00)
    # known carrier: 1 deg tilt along +x
    k0 = 2 * np.pi / LAM
    kx = k0 * np.sin(np.deg2rad(1.0))
    Et = (env * np.exp(1j * kx * X)).astype(complex)
    Ft, dg = apply_real_lens_traced_multibranch(
        Et, input_carrier='auto', return_diagnostics=True, **kw)
    kx_hat, ky_hat = dg['input_carrier']
    assert abs(kx_hat / kx - 1.0) < 1e-3
    assert abs(ky_hat) < 1e-3 * kx
    # chief-ray oracle: trace the tilted axial ray to the output plane
    L0 = kx / k0
    rays = rt.RayBundle(
        x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
        L=np.array([L0]), M=np.zeros(1),
        N=np.array([np.sqrt(1.0 - L0 ** 2)]),
        wavelength=LAM, alive=np.ones(1, bool), opd=np.zeros(1))
    ex = rt.trace(rays, rt.surfaces_from_prescription(presc),
                  LAM).image_rays
    x_pred = float(ex.x[0] + d / ex.N[0] * ex.L[0])
    I0 = np.abs(F0) ** 2
    It = np.abs(Ft) ** 2
    cx = float((It * X).sum() / It.sum()) - float((I0 * X).sum() / I0.sum())
    cy = float((It * Y).sum() / It.sum()) - float((I0 * Y).sum() / I0.sum())
    assert abs(cx / x_pred - 1.0) < 0.05
    assert abs(cy) < 2e-6
    assert abs(It.sum() / I0.sum() - 1.0) < 0.02


def test_ludwig_fold_exact_and_finite_through_caustic():
    """The Ludwig uniform fold formula (ray-native inputs: branch eikonals +
    COMPLEX amplitudes incl. their Maslov phases) reproduces the exact
    cubic-phase integral to machine precision on the bright side AND stays
    finite/exact through the caustic where the branch amplitudes diverge."""
    from scipy.special import airy

    from lumenairy.elements._lens_traced_multibranch import ludwig_fold

    for k in (300.0, 2000.0):
        for c in (0.2, 0.05, 1e-3, 1e-6):
            exact = 2 * np.pi * k ** (-1.0 / 3.0) * airy(
                -(k ** (2.0 / 3.0)) * c)[0]
            A = np.sqrt(2 * np.pi / (k * 2 * np.sqrt(c)))  # diverges as c->0
            Sp = 2.0 / 3.0 * c ** 1.5
            u = ludwig_fold(k, Sp, -Sp, A * np.exp(-1j * np.pi / 4),
                            A * np.exp(+1j * np.pi / 4))
            assert _relerr(u, exact) < 1e-11
            assert np.isfinite(u)


# ==========================================================================
# Adaptive delaminating Levin engine (lumenairy._math.levin)
# ==========================================================================
@pytest.mark.slow      # v5.21.0 release CI forensics: at tol=1e-8 /
                       # max_depth=16 this ran 6.6 s locally but >20 min on
                       # BOTH CI runner classes without completing -- the
                       # quadtree's parent-vs-4-child accept test at 1e-8
                       # sits at the FP-agreement floor for boxes straddling
                       # the fold line, so a runner whose libm rounds
                       # differently refines the whole caustic band to full
                       # depth (~2^depth boxes).  Gate at tol=1e-7 (10x off
                       # the floor) with max_depth=12 (bounded worst case);
                       # the engine's deeper 1e-8+ convergence is a local /
                       # full-suite property, not a CI gate.
def test_levin2d_uniform_through_fold_with_rigorous_bound():
    """The 2-D Levin engine integrates a fold-caustic phase (two coalescing
    stationary points) uniformly through mu -> 0, honoring its returned
    rigorous residual bound -- where stationary-phase methods diverge."""
    from lumenairy._math.levin import levin1d_adaptive, levin2d
    w2, lam, C = 60.0, 1.3, 2.0
    x0, y0 = 0.37, 0.63

    def one(t):
        return np.ones_like(t)

    def gxf(x):
        return w2 * 0.5 * lam * (x - x0) ** 2

    def gxp(x):
        return w2 * lam * (x - x0)

    Ix = levin1d_adaptive(gxf, gxp, one, -3.0, 3.0, tol=1e-13)
    for mu in (0.05, 0.0):                      # bright side AND on-caustic
        def gyf(y):
            return w2 * (C * (y - y0) ** 3 / 6.0 + 0.5 * mu * (y - y0) ** 2)

        def gyp(y):
            return w2 * (0.5 * C * (y - y0) ** 2 + mu * (y - y0))

        Iy = levin1d_adaptive(gyf, gyp, one, -3.0, 3.0, tol=1e-13)
        exact = Ix * Iy
        val, nb, bound = levin2d(
            lambda x, y: gxf(x) + gyf(y),
            lambda x, y: gxp(x) + 0 * y,
            lambda x, y: gyp(y) + 0 * x,
            lambda x, y: np.ones_like(x),
            (-3, 3, -3, 3), tol=1e-7, k=7, max_depth=12)
        assert bound <= 1e-7 * 1.01             # budget met
        assert abs(val - exact) < 5e-8          # and honest
        assert abs(val - exact) <= bound * 1.01  # rigorous bound holds


@pytest.mark.slow      # Levin ROI vs converged quadrature reference --
                       # minutes-scale on CI; keep the fast gate lean
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_maslov_levin_matches_quadrature_roi():
    """integration_method='levin' reproduces the converged uniform quadrature
    on a small ROI of a stopped-down singlet chart -- a PLUMBING check of the
    integrand mapping (phase / derivatives / |det ds1/du| / Tukey / measure);
    the engine's own accuracy is covered by the canonical fold test.  ROI-
    sized because the per-pixel adaptive engine costs O(10 s) per pixel."""
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    presc = {'aperture_diameter': 2e-3, 'surfaces': [
        {'radius': 25e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': float('inf'), 'glass_before': 'N-BK7',
         'glass_after': 'air'},
        {'radius': float('inf'), 'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [3e-3, 2e-3, 0.0]}
    N, dx = 16, 18e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (0.6e-3) ** 2).astype(np.complex128)
    kw = dict(prescription=presc, wavelength=LAM, dx=dx, roi=(0.0, 0.0, dx))
    Fq = apply_real_lens_maslov(E, integration_method='quadrature',
                                n_v2=256, **kw)
    Fl = apply_real_lens_maslov(E, integration_method='levin',
                                levin_tol=1e-2, **kw)
    assert Fl.shape == Fq.shape
    assert _relerr(Fl, Fq) < 0.1               # mapping correct; ref-limited


def test_pearcey_matches_exact_cusp_value_and_symmetry():
    """The Pearcey (cusp) evaluator matches the exact cusp value
    P(0,0)=1/2 Gamma(1/4) exp(i pi/8), is even in y, and agrees with an
    independent contour-rotated quadrature at y=0."""
    from math import gamma

    from lumenairy.elements.lenses_maslov import pearcey

    p00 = pearcey(0.0, 0.0)
    exact00 = 0.5 * gamma(0.25) * np.exp(1j * np.pi / 8)
    assert _relerr(p00, exact00) < 1e-12
    assert abs(abs(p00) - 1.8128043) < 1e-5
    assert abs(np.degrees(np.angle(p00)) - 22.5) < 1e-4

    # even in y
    for x, y in [(1.0, 2.0), (-2.0, 1.5), (3.0, -1.0)]:
        assert _relerr(pearcey(x, y), pearcey(x, -y)) < 1e-13

    # vs independent contour quadrature at y=0
    def quad_y0(x, R=7.0, n=400000):
        s = np.linspace(0.0, R, n)
        integ = np.exp(-s ** 4 + 1j * x * (s ** 2) * np.exp(1j * np.pi / 4))
        return 2.0 * np.exp(1j * np.pi / 8) * np.trapezoid(integ, s)

    for x in (0.0, 1.5, -2.0, 3.0):
        assert _relerr(pearcey(x, 0.0), quad_y0(x)) < 1e-6
