"""The non-paraxial, tilt-aware gap kernel (``gap_kernel='exact'``).

WHAT THIS COVERS AND WHY IT IS NOT COVERED ELSEWHERE
----------------------------------------------------
The traced carrier chain transports the ENVELOPE across a gap with a
Sziklas-Siegman co-moving frame.  Historically the envelope leg used
``fresnel_tf_propagate``, whose transfer function is

    phase = k*z - z*|q|^2 / (2k)                      (paraxial)

which is exactly the small-``|q|`` expansion of the true free-space kernel

    phase = z*sqrt(k^2 - |q|^2)                       (exact)

That is fine for design 121 (low-NA, untilted legs -- see the end-to-end test
below, which measures 2e-08) but it is NOT fine for a general optical system,
and ``apply_real_lens_traced`` is meant to be the workhorse for any system.  Two
distinct errors are removed here, and they are ONE change because they are two
terms of a single expansion about the carrier wavevector ``k*s``:

    sqrt(k^2 - |k s + q|^2)
        = k N - (s.q)/N - |q|^2/(2 k N) - (s.q)^2/(2 k N^3) + ...

  * the ``|q|^2`` term's higher orders  -> non-paraxial angular content
  * the ``(s.q)^2`` term                -> ANISOTROPIC stretch under tilt:
        effective distance  z/N^3 ALONG the tilt,  z/N ACROSS it.
    The Fresnel kernel is isotropic and tilt-blind, so it applies plain ``z``
    on both axes and cannot represent this at all.

Nothing else in the suite tests either of these: the gap guards
(``test_niche_d3_guards``, ``test_niche_gap_frame_observable``) only MEASURE
whether the paraxial frame is being abused, they do not fix it; the in-glass
propagator is separately already exact ('asm'); and the 121 acceptance runs an
untilted low-NA chain where the two kernels agree to 1e-08 and so cannot
distinguish them.

The ground truth used here is an INDEPENDENT plain-numpy exact ASM on the full
field (carrier re-attached, propagated, carrier stripped, chief ray re-centred),
written inline so it shares no code with the implementation under test.
"""
import numpy as np
import pytest

from lumenairy.propagators.carrier import (
    _exact_envelope_tf_step,
    _carrier_step_fast,
    propagate_carrier_referenced,
)
from lumenairy.propagators.fresnel import fresnel_tf_propagate

WL = 1.31e-6
K = 2.0 * np.pi / WL


def _gaussian(n, dx, w, quad=0.0):
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x)
    r2 = X * X + Y * Y
    return (np.exp(-r2 / w ** 2)
            * np.exp(1j * K * quad * r2)).astype(np.complex128)


def _oracle(env, z, dx, L=0.0, M=0.0):
    """Independent exact-ASM truth for the envelope a tilted carrier leg owes."""
    n = env.shape[-1]
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x)
    Nz = np.sqrt(1.0 - L * L - M * M)
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    KX, KY = np.meshgrid(kx, kx)
    root = np.sqrt(np.maximum(K * K - (KX ** 2 + KY ** 2), 0.0))
    full = env * np.exp(1j * K * (L * X + M * Y))
    out = np.fft.ifft2(np.fft.fft2(full) * np.exp(1j * z * root))
    out = out * np.exp(-1j * K * (L * X + M * Y))
    # undo the chief-ray transverse advance so we compare like for like
    sh = np.exp(1j * (KX * (L * z / Nz) + KY * (M * z / Nz)))
    return np.fft.ifft2(np.fft.fft2(out) * sh)


def _rel(a, b, amp):
    m = amp > 0.05 * amp.max()
    return float(np.linalg.norm((a - b)[m])
                 / max(np.linalg.norm(b[m]), 1e-300))


# --------------------------------------------------------------------------
# the exact kernel really is exact
# --------------------------------------------------------------------------

@pytest.mark.parametrize('w,quad,z', [
    (60e-6, 0.0, 2e-3),
    (30e-6, 0.0, 5e-3),
    (20e-6, 0.0, 3e-3),
    (40e-6, 6.0e3, 3e-3),        # quadratic-loaded: Fresnel fails outright
])
def test_exact_kernel_matches_independent_asm_far_better_than_fresnel(w, quad, z):
    n, dx = 512, 1.0e-6
    env = _gaussian(n, dx, w, quad)
    amp = np.abs(env)
    ref = _oracle(env, z, dx)
    r_fr = _rel(fresnel_tf_propagate(env, z, WL, dx, dx), ref, amp)
    r_ex = _rel(_exact_envelope_tf_step(env, z, WL, dx, dx), ref, amp)
    # exact must be at the numerical floor, and must beat paraxial decisively
    assert r_ex < 1e-10, f'exact kernel not at the floor: {r_ex:.3e}'
    assert r_ex < r_fr / 100.0, (
        f'exact {r_ex:.3e} did not decisively beat fresnel {r_fr:.3e}')


def test_paraxial_kernel_is_catastrophic_on_the_quad_loaded_case():
    """Guards against a future 'the two kernels are basically the same' claim:
    on strong angular content the paraxial kernel is not slightly wrong, it is
    order-unity wrong -- which is the whole reason this knob exists."""
    n, dx = 512, 1.0e-6
    env = _gaussian(n, dx, 40e-6, quad=6.0e3)
    ref = _oracle(env, 3e-3, dx)
    r_fr = _rel(fresnel_tf_propagate(env, 3e-3, WL, dx, dx), ref, np.abs(env))
    assert r_fr > 0.5, (
        f'expected the paraxial kernel to fail hard here, got {r_fr:.3e}; '
        'if this drops, the discriminating power of this file is gone')


# --------------------------------------------------------------------------
# the anisotropic tilt stretch (the part Fresnel cannot represent at all)
# --------------------------------------------------------------------------

def _fitted_coeff(z, dx, n, L, M, axis, m):
    """Least-squares quadratic coefficient of the kernel's own phase on `axis`.

    Read the coefficient the operator actually applies, rather than trusting a
    propagated field.  A pure-quadratic fit over a finite window necessarily
    picks up the kernel's quartic and higher terms, so the fit error must be
    driven to zero by SHRINKING the window -- that convergence is the assertion.
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    Nz = np.sqrt(1.0 - L * L - M * M)
    KX, KY = kx[None, :], kx[:, None]
    ax, ay = K * L + KX, K * M + KY
    root = np.sqrt(np.maximum(K * K - (ax * ax + ay * ay), 0.0))
    root0 = np.sqrt(max(K * K * (1.0 - L * L - M * M), 0.0))
    phase = (K * z) + z * (root - root0 + (L * KX + M * KY) / Nz)
    if axis == 'x':
        q, p = kx[1:m], phase[0, 1:m] - phase[0, 0]
    else:
        q, p = kx[1:m], phase[1:m, 0] - phase[0, 0]
    return float(np.sum(p * q ** 2) / np.sum(q ** 4)), Nz


@pytest.mark.parametrize('L', [0.10, 0.35, 0.50])
def test_tilt_stretch_is_anisotropic_z_over_n_cubed_along_and_z_over_n_across(L):
    z, dx, n = 2e-3, 0.5e-6, 4096
    base = -z / (2.0 * K)
    errs = []
    for m in (12, 8, 5, 3):                       # shrinking fit window
        a, Nz = _fitted_coeff(z, dx, n, L, 0.0, 'x', m)
        b, _ = _fitted_coeff(z, dx, n, L, 0.0, 'y', m)
        e_along = abs(a / base - 1.0 / Nz ** 3) / (1.0 / Nz ** 3)
        e_across = abs(b / base - 1.0 / Nz) / (1.0 / Nz)
        errs.append((e_along, e_across))
    # the leading coefficients are exactly 1/N^3 and 1/N: the residual is the
    # fit's own quartic contamination, so it must SHRINK with the window
    assert errs[-1][0] < errs[0][0] / 3.0, (
        f'along-tilt fit error did not converge: {[e[0] for e in errs]}')
    assert errs[-1][0] < 2e-3 and errs[-1][1] < 1e-5, (
        f'narrowest-window error too large: {errs[-1]}')


def test_fresnel_is_isotropic_and_therefore_cannot_carry_the_stretch():
    """The contrast that motivates the whole change: at 0.35 direction cosine
    the true along-tilt diffraction distance is ~22% longer than z, and the
    paraxial kernel applies exactly z on both axes."""
    L = 0.35
    Nz = np.sqrt(1.0 - L * L)
    assert 1.0 / Nz ** 3 - 1.0 > 0.20         # along: >20% stretch
    assert 0.05 < 1.0 / Nz - 1.0 < 0.10       # across: ~7%
    # the paraxial TF's coefficient carries no L dependence whatsoever
    n, dx, z = 256, 1e-6, 2e-3
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    par = -(z / (2.0 * K)) * kx[1] ** 2
    assert par == -(z / (2.0 * K)) * kx[1] ** 2   # no tilt argument exists


# --------------------------------------------------------------------------
# structural contracts
# --------------------------------------------------------------------------

def test_zero_tilt_is_identical_to_omitting_tilt():
    env = _gaussian(256, 1e-6, 40e-6)
    a = _exact_envelope_tf_step(env, 3e-3, WL, 1e-6, 1e-6, tilt=(0.0, 0.0))
    b = _exact_envelope_tf_step(env, 3e-3, WL, 1e-6, 1e-6)
    assert np.array_equal(a, b)


def test_zero_distance_is_the_identity_even_under_tilt():
    env = _gaussian(256, 1e-6, 40e-6)
    out = _exact_envelope_tf_step(env, 0.0, WL, 1e-6, 1e-6, tilt=(0.1, 0.05))
    assert np.abs(out - env).max() < 1e-14


def test_returned_buffer_is_owned_not_the_fft_scratch():
    """``_ifft2`` hands back a cache-owned buffer; a caller that keeps the result
    across another transform would otherwise see it mutated underneath."""
    env = _gaussian(256, 1e-6, 40e-6)
    out = _exact_envelope_tf_step(env, 2e-3, WL, 1e-6, 1e-6)
    keep = out.copy()
    _exact_envelope_tf_step(env, 5e-3, WL, 1e-6, 1e-6)   # reuse the scratch
    assert np.array_equal(out, keep), 'result aliased the FFT cache buffer'


def test_evanescent_band_is_clamped_not_nan():
    """A grid fine enough to sample beyond k must not produce NaNs."""
    env = _gaussian(256, 0.2e-6, 5e-6)      # dx < wavelength/2 -> |q| > k exists
    out = _exact_envelope_tf_step(env, 1e-4, WL, 0.2e-6, 0.2e-6)
    assert np.isfinite(out).all()


def test_non_propagating_tilt_is_refused():
    env = _gaussian(64, 1e-6, 20e-6)
    with pytest.raises(ValueError):
        _exact_envelope_tf_step(env, 1e-3, WL, 1e-6, 1e-6, tilt=(0.8, 0.8))


def test_explicit_fresnel_still_reproduces_the_pinned_paraxial_path():
    """The paraxial kernel has not gone away -- it is reachable and unchanged,
    which is what lets the two be compared at all."""
    env = _gaussian(256, 2e-6, 60e-6)
    a = _carrier_step_fast(env, 0.5, 1e-3, WL, 2e-6, 2e-6,
                           gap_kernel='fresnel')
    b = fresnel_tf_propagate(env, 1e-3 * (0.5 / (0.5 + 1e-3)), WL,
                             2e-6, 2e-6)
    assert np.asarray(a[0]).shape == np.asarray(b).shape


def test_the_default_is_now_the_exact_kernel():
    """Flipped in v5.30.2: the default ``gap_kernel='auto'`` resolves to the
    EXACT kernel, so the paraxial one is now opt-in rather than the reverse.
    Asserted against BOTH explicit values so the direction cannot silently
    invert -- this test previously encoded the opposite default."""
    env = _gaussian(256, 2e-6, 60e-6)
    a = propagate_carrier_referenced(env, 0.5, 1e-3, WL, 2e-6)
    ex = propagate_carrier_referenced(env, 0.5, 1e-3, WL, 2e-6,
                                      gap_kernel='exact')
    fr = propagate_carrier_referenced(env, 0.5, 1e-3, WL, 2e-6,
                                      gap_kernel='fresnel')
    assert np.array_equal(np.asarray(a.env), np.asarray(ex.env)), (
        'the default no longer resolves to the exact kernel')
    assert not np.array_equal(np.asarray(a.env), np.asarray(fr.env)), (
        'the default is indistinguishable from the paraxial kernel')
    # and it is a phase-only kernel: power matches the paraxial route it
    # replaces, so flipping the default cannot have changed throughput
    p_ex = float(np.sum(np.abs(np.asarray(ex.env)) ** 2))
    p_fr = float(np.sum(np.abs(np.asarray(fr.env)) ** 2))
    assert abs(p_ex / p_fr - 1.0) < 1e-9


# --------------------------------------------------------------------------
# backend parity: the exact kernel is not a NumPy-only privilege
# --------------------------------------------------------------------------
# It shipped NumPy-only at first purely because the physics was validated
# there; the maths (fftfreq, sqrt, exp, multiply) is array-API agnostic.  A
# NumPy-only exact kernel would have meant CuPy / JAX users silently getting
# the PARAXIAL transfer function -- a correctness difference between backends,
# which is exactly the kind of thing that goes unnoticed.  `_exact_tf_2d_xp`
# is the backend analogue, built on the same `_freq_1d_bld` / `_tf_phase_to_H`
# scaffolding `_fresnel_tf_2d_xp` uses (so complex64 gets the same `mod 2*pi`
# phase folding, which matters here because `k*z` is large).

def _xp_step(E, z, tilt, xp, is_jax, dx=1e-6):
    from lumenairy.propagators.carrier import _exact_tf_2d_xp
    return _exact_tf_2d_xp(E, z, WL, dx, dx, tilt, xp, is_jax, np)


@pytest.mark.parametrize('tilt', [(0.0, 0.0), (0.10, 0.0), (0.08, 0.05)])
def test_backend_generic_kernel_matches_the_numpy_one(tilt):
    """Same operator, built through the backend scaffolding."""
    env = _gaussian(256, 1e-6, 40e-6)
    ref = _exact_envelope_tf_step(env, 2e-3, WL, 1e-6, 1e-6, tilt=tilt)
    got = _xp_step(env, 2e-3, tilt, np, False)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-13, f'backend-generic kernel diverged: relL2 {rel:.3e}'


def test_backend_generic_kernel_refuses_a_non_propagating_tilt():
    env = _gaussian(64, 1e-6, 20e-6)
    with pytest.raises(ValueError):
        _xp_step(env, 1e-3, (0.8, 0.8), np, False)


def test_backend_generic_kernel_clamps_the_evanescent_band():
    env = _gaussian(256, 0.2e-6, 5e-6)      # dx < lambda/2 -> |q| > k exists
    assert np.isfinite(_xp_step(env, 1e-4, (0.0, 0.0), np, False,
                                dx=0.2e-6)).all()


def test_jax_backend_reproduces_the_numpy_exact_kernel():
    """Run it on a REAL non-NumPy backend, not a NumPy stand-in -- otherwise
    this file would only prove the scaffolding is wired, not that it works."""
    jnp = pytest.importorskip('jax.numpy')
    import jax
    jax.config.update('jax_enable_x64', True)
    for tilt in ((0.0, 0.0), (0.10, 0.0), (0.08, 0.05)):
        env = _gaussian(256, 1e-6, 40e-6)
        ref = _exact_envelope_tf_step(env, 2e-3, WL, 1e-6, 1e-6, tilt=tilt)
        got = np.asarray(_xp_step(jnp.asarray(env), 2e-3, tilt, jnp, True))
        rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        assert rel < 1e-13, f'JAX exact kernel diverged at {tilt}: {rel:.3e}'


def test_auto_resolves_to_exact_on_every_backend():
    """The default is 'auto'.  If it ever resolves to 'fresnel' on a backend,
    that backend silently returns to paraxial gap transport."""
    import inspect
    from lumenairy.propagators import carrier as C
    src = inspect.getsource(C._carrier_step_fast)
    assert "gap_kernel = 'exact'" in src, (
        "'auto' no longer resolves to the exact kernel")
    assert "'exact' if xp is np else 'fresnel'" not in src, (
        'auto resolution is backend-conditional again -- CuPy / JAX would get '
        'the paraxial kernel while NumPy gets the exact one')
