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
    _carrier_step_fast,
    _exact_envelope_tf_step,
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


# --------------------------------------------------------------------------
# D4: the gap_kernel vocabulary is STRICT -- a typo cannot buy back Fresnel
# --------------------------------------------------------------------------
# Fail-before (REVIEW_TRACED_EXACT_2026_08_05 D4, re-measured 2026-08-06 on the
# working tree): ``propagate_carrier_referenced`` resolved the kernel with an
# if/elif chain whose last arm was an unguarded catch-all, so
#
#     'exsct' -> FRESNEL (dist_to_fresnel = 0.000e+00)
#     'EXACT' -> FRESNEL      None -> FRESNEL      1 / '' -> FRESNEL
#
# and ``carrier_referenced_focus_readout`` had no validation at all.  Silently
# running the paraxial kernel is the one outcome this whole campaign exists to
# prevent, so every entry point that accepts the knob must refuse a value it
# does not recognise.

_BAD_GAP_KERNELS = ('exsct', 'EXACT', 'Fresnel', 'paraxial', '', None, 1,
                    ('exact',))


def _gap_kernel_entry_points():
    """Every PUBLIC name in ``carrier.__all__`` whose signature takes
    ``gap_kernel``.  Discovered, not listed, so a new entry point that grows
    the knob is caught here rather than shipping unvalidated."""
    import inspect

    from lumenairy.propagators import carrier as C
    found = []
    for name in C.__all__:
        obj = getattr(C, name)
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):        # pragma: no cover - C callables
            continue
        if 'gap_kernel' in sig.parameters:
            found.append(name)
    return sorted(found)


def _minimal_call(name, gap_kernel):
    """Smallest legal call of ``name`` that reaches its gap_kernel gate."""
    from lumenairy.propagators import carrier as C
    env = _gaussian(64, 2e-6, 20e-6)
    fn = getattr(C, name)
    if name == 'propagate_carrier_referenced':
        return fn(env, 0.5, 1e-3, WL, 2e-6, gap_kernel=gap_kernel)
    if name == 'carrier_referenced_focus_readout':
        return fn(env, -0.05, 0.04, WL, 2e-6, dx_out=0.4e-6, N_out=32,
                  gap_kernel=gap_kernel)
    if name == 'propagate_traced_carrier_chain':
        return fn(env, [], WL, 2e-6, gap_kernel=gap_kernel)
    if name == 'propagate_traced_carrier_chain_multi':
        return fn([], [], WL, 2e-6,
                  output_grid={'dx_out': 0.4e-6, 'N_out': 32},
                  gap_kernel=gap_kernel)
    raise AssertionError(                       # pragma: no cover
        f'{name} accepts gap_kernel but has no minimal call here -- add one '
        f'(and check it validates the knob) rather than deleting this arm')


def test_the_gap_kernel_entry_points_are_the_ones_we_think_they_are():
    """Premise guard for the walk below.  If a new public function grows a
    ``gap_kernel`` argument this fails until it is added to ``_minimal_call``
    -- which is the point: the D4 defect was an entry point nobody checked."""
    assert _gap_kernel_entry_points() == [
        'carrier_referenced_focus_readout',
        'propagate_carrier_referenced',
        'propagate_traced_carrier_chain',
        'propagate_traced_carrier_chain_multi',
    ]


@pytest.mark.parametrize('name', _gap_kernel_entry_points())
@pytest.mark.parametrize('bad', _BAD_GAP_KERNELS)
def test_every_public_entry_point_refuses_an_unrecognised_gap_kernel(name, bad):
    with pytest.raises(ValueError) as ei:
        _minimal_call(name, bad)
    msg = str(ei.value)
    assert 'gap_kernel' in msg, f'{name}: message does not name the knob: {msg}'
    for kind in ("'auto'", "'exact'", "'fresnel'"):
        assert kind in msg, (
            f'{name}: message does not name the accepted value {kind}: {msg}')


@pytest.mark.parametrize('name', _gap_kernel_entry_points())
@pytest.mark.parametrize('good', ['auto', 'exact', 'fresnel'])
def test_every_public_entry_point_still_accepts_the_whole_vocabulary(name, good):
    """The gate must not be over-strict: all three documented values survive it.
    (Some minimal calls raise for unrelated reasons -- an empty ``groups`` --
    so the assertion is that nothing gap_kernel-shaped is refused.)"""
    try:
        _minimal_call(name, good)
    except ValueError as exc:
        assert 'gap_kernel' not in str(exc), (
            f'{name} rejected the documented value {good!r}: {exc}')


def test_the_resolver_itself_refuses_rather_than_falling_through():
    """``_carrier_step_fast`` is the single site every carrier leg funnels
    through.  Gating it there means a future entry point that forgets its own
    check still cannot reach the paraxial arm by accident."""
    import inspect

    from lumenairy.propagators import carrier as C
    env = _gaussian(64, 2e-6, 20e-6)
    for bad in ('exsct', 'EXACT', None):
        with pytest.raises(ValueError):
            _carrier_step_fast(env, 0.5, 1e-3, WL, 2e-6, 2e-6, gap_kernel=bad)
    src = inspect.getsource(C._carrier_step_fast)
    assert '_check_gap_kernel(' in src, (
        'the resolver no longer validates its own argument -- an unrecognised '
        'value can fall through the elif chain to the paraxial kernel again')
    assert C._GAP_KERNELS == ('auto', 'exact', 'fresnel')


def test_none_is_refused_not_treated_as_the_default():
    """``None`` is what a forgotten/unset variable looks like, and the
    documented default is the STRING 'auto'.  Mapping None to a default would
    put back the silent path; it is refused, and the message says so."""
    env = _gaussian(64, 2e-6, 20e-6)
    with pytest.raises(ValueError, match='gap_kernel'):
        propagate_carrier_referenced(env, 0.5, 1e-3, WL, 2e-6, gap_kernel=None)


def test_the_sibling_mode_knobs_are_refused_up_front_by_the_multi_orchestrator():
    """Same defect class, checked on the knobs D4 names as siblings.
    ``final_leg`` / ``carrier_reference`` used to be validated ONLY inside the
    per-congruence chain call -- i.e. after this orchestrator had already sized
    its memory clamp from ``final_leg != 'paraxial'`` and, with
    ``congruence_workers > 1``, after the raise had to be marshalled out of a
    worker process."""
    from lumenairy.propagators.carrier import propagate_traced_carrier_chain_multi
    og = {'dx_out': 0.4e-6, 'N_out': 32}
    with pytest.raises(ValueError, match='final_leg'):
        propagate_traced_carrier_chain_multi([], [], WL, 2e-6, output_grid=og,
                                             final_leg='paraxail')
    with pytest.raises(ValueError, match='final_leg'):
        propagate_traced_carrier_chain_multi([], [], WL, 2e-6, output_grid=og,
                                             final_leg=None)
    with pytest.raises(ValueError, match='carrier_reference'):
        propagate_traced_carrier_chain_multi([], [], WL, 2e-6, output_grid=og,
                                             carrier_reference='Sphere')


# --------------------------------------------------------------------------
# D7: the frequency grid is fftfreq for BOTH parities of N
# --------------------------------------------------------------------------
# ``_freq_1d_bld`` / ``_freq_sq_1d_bld`` build the CENTRED axis and every
# caller un-shifts it with ``ifftshift``, so the bins must be the integers
# ``j - N//2``.  They used ``- N / 2``, which is the same number for even N but
# HALF-INTEGER for odd N, and ``ifftshift`` of a half-integer axis is not
# ``fftfreq``:
#
#     N = 5:  ifftshift(old) = [-0.1, 0.1, 0.3, -0.5, -0.3]
#             fftfreq(5)     = [ 0.0, 0.2, 0.4, -0.4, -0.2]
#
# Fail-before, both 2-D builders against their NumPy twins (2026-08-06):
#     _exact_tf_2d_xp   N=65  relL2 1.239e+00   N=127 relL2 7.209e-01
#     _fresnel_tf_2d_xp N=65  relL2 1.239e+00   N=127 relL2 7.189e-01
# against 4e-16 at N = 64 / 128.  Nothing enforces even grids, and no test in
# this file used an odd N -- so the backend-parity guarantee this whole
# function exists for was false on half the possible grids.

_PARITY_NS = (64, 65, 127, 128, 1)


@pytest.mark.parametrize('N', _PARITY_NS)
@pytest.mark.parametrize('d', [1.0, 1e-6])
def test_freq_grids_reproduce_fftfreq_for_both_parities(N, d):
    from lumenairy.propagators.carrier import _freq_1d_bld, _freq_sq_1d_bld
    ref = np.fft.fftfreq(N, d=d)
    got = np.fft.ifftshift(_freq_1d_bld(N, d, np))
    got_sq = np.fft.ifftshift(_freq_sq_1d_bld(N, d, np))
    # (a) the BIN MAPPING is exact -- this is the part that was wrong.  Compare
    # in integer space (f * N * d), where both sides are whole numbers.
    assert np.array_equal(np.rint(got * (N * d) / (2.0 * np.pi)),
                          np.rint(ref * (N * d))), (
        f'N={N}: bin mapping is not fftfreq\n  got {got}\n  ref {2*np.pi*ref}')
    # (b) and the VALUES agree to a couple of ulp.  Not bit-exact by
    # construction: fftfreq multiplies by 1/(N*d), the builders divide by
    # (N*d), which differ in the last place whenever N*d is not a power of two.
    # Making them bit-equal would mean switching to the reciprocal multiply,
    # which would perturb the validated EVEN-N path -- so it is deliberately
    # not done.
    scale = max(float(np.abs(2.0 * np.pi * ref).max()), np.finfo(float).tiny)
    assert float(np.abs(got - 2.0 * np.pi * ref).max()) <= 4.0 * np.spacing(scale)
    assert float(np.abs(got_sq - (2.0 * np.pi * ref) ** 2).max()) <= \
        4.0 * np.spacing(scale ** 2)


@pytest.mark.parametrize('N', _PARITY_NS)
def test_the_centred_axis_is_the_integer_bin_formula_exactly(N):
    """The invariant that makes the even-N path BIT-IDENTICAL across this fix:
    the builders are exactly ``(arange(N) - N//2)/(N*d)``, and for even N
    ``N//2 == N/2``, so no even-N value moved.  Pinned so a future 'tidy-up'
    back to ``N/2`` (or to a reciprocal multiply) is caught."""
    from lumenairy.propagators.carrier import _freq_1d_bld, _freq_sq_1d_bld
    d = 1.31e-6
    f = (np.arange(N, dtype=np.float64) - (N // 2)) / (N * d)
    assert np.array_equal(_freq_1d_bld(N, d, np), 2.0 * np.pi * f)
    assert np.array_equal(_freq_sq_1d_bld(N, d, np), (2.0 * np.pi * f) ** 2)
    if N % 2 == 0:
        f_old = (np.arange(N, dtype=np.float64) - N / 2) / (N * d)
        assert np.array_equal(_freq_1d_bld(N, d, np), 2.0 * np.pi * f_old), (
            'the EVEN-N axis moved -- the validated path is not bit-identical')


@pytest.mark.parametrize('N', [65, 127])
@pytest.mark.parametrize('tilt', [(0.0, 0.0), (0.05, 0.02)])
def test_backend_generic_exact_kernel_matches_the_numpy_one_at_odd_n(N, tilt):
    env = _gaussian(N, 1e-6, 20e-6)
    ref = _exact_envelope_tf_step(env, 2e-3, WL, 1e-6, 1e-6, tilt=tilt)
    got = _xp_step(env, 2e-3, tilt, np, False)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-13, f'odd-N backend kernel diverged at N={N}: {rel:.3e}'


@pytest.mark.parametrize('N', [65, 127])
def test_backend_generic_fresnel_kernel_matches_the_numpy_one_at_odd_n(N):
    """The SAME defect lived in ``_fresnel_tf_2d_xp`` (it is where
    ``_freq_1d_bld`` was copied from), so it is pinned here too."""
    from lumenairy.propagators.carrier import _fresnel_tf_2d_xp
    env = _gaussian(N, 1e-6, 20e-6)
    ref = fresnel_tf_propagate(env, 2e-3, WL, 1e-6, 1e-6)
    got = _fresnel_tf_2d_xp(env, 2e-3, WL, 1e-6, 1e-6, np, False, np)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 1e-13, f'odd-N backend Fresnel kernel diverged: {rel:.3e}'


@pytest.mark.parametrize('N', [65, 127])
def test_odd_n_exact_kernels_both_match_the_independent_asm_oracle(N):
    """The adjudicating arm: not just 'the two implementations agree' but
    'both agree with an oracle that shares no code with either'.  Measured
    2026-08-06: numpy-vs-oracle 2.744e-14 (N=65) / 1.386e-12 (N=127), and the
    backend build now lands on the same numbers (it was 1.239 / 0.719)."""
    env = _gaussian(N, 1e-6, 20e-6)
    ref = _oracle(env, 2e-3, 1e-6)
    amp = np.abs(env)
    r_np = _rel(_exact_envelope_tf_step(env, 2e-3, WL, 1e-6, 1e-6), ref, amp)
    r_xp = _rel(_xp_step(env, 2e-3, (0.0, 0.0), np, False), ref, amp)
    assert r_np < 1e-10, f'NumPy exact kernel vs oracle at N={N}: {r_np:.3e}'
    assert r_xp < 1e-10, f'backend exact kernel vs oracle at N={N}: {r_xp:.3e}'


# --------------------------------------------------------------------------
# COMPOSITION ACROSS A SPLIT LEG -- and why the exact kernel cannot have it
# --------------------------------------------------------------------------
# Found 2026-08-06 as 6 failures in ``test_niche_d4_dgrating``: propagating a
# leg in one step and in two agreed to 1e-11 under the paraxial default and
# disagrees by up to 9.4e-03 under the exact one.  It is not a bug, and it is
# not fixable; it is a theorem, and these tests pin both halves of it.
#
# WHY THE PARAXIAL KERNEL COMPOSES.  Split z into z1 + z2 about a carrier R.
# Leg 2 runs on the MAGNIFIED co-moving grid (pitch m1*dx, m1 = (R+z1)/R), so
# the same FFT bin is a factor m1 LOWER physical frequency there: the composed
# phase at bin q is  phi(z_eff1, q) + phi(z_eff2, q/m1).  The reduced distances
# telescope exactly,
#
#     z_eff1 + z_eff2 / m1^2 == z_eff        (verified below to 0 / 3e-16)
#
# so any kernel whose only q dependence is q^2 -- i.e. the paraxial one --
# composes exactly.
#
# WHY NO EXACT KERNEL CAN.  Composition for every split requires
#
#     phi(t1, q) + phi(t2, q/m) = phi(t1 + t2/m^2, q)   for all m,
#
# and substituting s = t*q^2 turns that into phi~(s1) + phi~(s2) = phi~(s1+s2)
# -- Cauchy's equation -- so phi~ is LINEAR in z_eff*q^2.  A kernel that
# composes across Sziklas-Siegman splits is therefore necessarily paraxial.
# The exact kernel z*sqrt(k^2-q^2) = k z - z q^2/2k - z q^4/8k^3 - ... carries
# z*q^4, which is not a function of z*q^2, so it composes only when m = 1.
#
# WHAT THIS COSTS, MEASURED.  The composition residual is bounded BELOW the
# non-paraxial correction the exact kernel applies (test below), and both the
# split and the single exact routes beat both paraxial routes against an exact
# full-field oracle (the test after that, which also closes review D10 -- the
# exact kernel had never been adjudicated inside a SCALED step at all).

_SPLIT_Z1 = 51.5393e-3        # design 121's DOE gap_before, to scale
_SPLIT_Z2 = 7.0e-3            # ... and its gap_after
_SPLIT_RS = [np.inf, 703.5912, -0.2, -0.045, -0.030, -0.010, -0.003]


def _split_env(n=256, dx=20e-6, w=1.5e-3):
    x = (np.arange(n) - n // 2) * dx
    return np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / w ** 2
                  ).astype(np.complex128), dx


def _one_and_two(R, kernel, z1=_SPLIT_Z1, z2=_SPLIT_Z2, n=256, dx=20e-6,
                 w=1.5e-3):
    import warnings
    env, dx = _split_env(n, dx, w)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        one = propagate_carrier_referenced(env, R, z1 + z2, WL, dx,
                                           gap_kernel=kernel)
        b1 = propagate_carrier_referenced(env, R, z1, WL, dx,
                                          gap_kernel=kernel)
        two = propagate_carrier_referenced(b1.env, b1.R, z2, WL, b1.dx,
                                           gap_kernel=kernel)
    return env, one, two


def _maxrel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-300))


@pytest.mark.parametrize('R', _SPLIT_RS)
def test_the_reduced_distances_telescope_exactly(R):
    """The algebraic identity the paraxial composition rests on."""
    if not np.isfinite(R):
        pytest.skip('m == 1 identically for a collimated carrier')
    z1, z2 = _SPLIT_Z1, _SPLIT_Z2
    m1 = (R + z1) / R
    ze1 = z1 * R / (R + z1)
    ze2 = z2 * (R + z1) / (R + z1 + z2)
    ze = (z1 + z2) * R / (R + z1 + z2)
    assert abs((ze1 + ze2 / (m1 * m1)) / ze - 1.0) < 1e-14


@pytest.mark.parametrize('R', _SPLIT_RS)
def test_the_paraxial_kernel_composes_across_a_split_leg(R):
    """The theorem half that HOLDS: with the paraxial kernel a split leg is
    inert to the FFT round-trip floor, at every carrier including the ones
    whose focus falls inside the leg (which route through the bridge)."""
    _e, one, two = _one_and_two(R, 'fresnel')
    assert one.dx == pytest.approx(two.dx, rel=1e-12)
    assert _maxrel(one.env, two.env) < 1e-9


@pytest.mark.parametrize('R', [-0.2, -0.5, 0.3])
def test_the_exact_split_residual_is_exactly_the_predicted_kernel_mismatch(R):
    """The theorem half that FAILS, pinned to its own arithmetic: the whole
    split-vs-single field difference is the composed-minus-single TOTAL phase
    (transfer function plus Sziklas-Siegman piston), predicted here from the
    closed form.  Agreement means no piston / amplitude / pitch bookkeeping is
    involved -- it is the kernel and nothing else.  Non-crossing legs only, so
    no bridge re-grid enters."""
    n, dx, w = 256, 20e-6, 1.5e-3
    z1, z2 = _SPLIT_Z1, _SPLIT_Z2
    env, _ = _split_env(n, dx, w)
    m1 = (R + z1) / R
    m = (R + z1 + z2) / R
    assert m1 > 0, 'this case must not cross the focus'
    ze1 = z1 * R / (R + z1)
    ze2 = z2 * (R + z1) / (R + z1 + z2)
    ze = (z1 + z2) * R / (R + z1 + z2)
    p1 = K * z1 * z1 / (R + z1)
    p2 = K * z2 * z2 / (R + z1 + z2)
    p = K * (z1 + z2) ** 2 / (R + z1 + z2)
    q = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    Q2 = q[None, :] ** 2 + q[:, None] ** 2

    def _phi(t, q2):
        return t * np.sqrt(np.maximum(K * K - q2, 0.0))

    dphi = (_phi(ze1, Q2) + p1 + _phi(ze2, Q2 / (m1 * m1)) + p2
            - (_phi(ze, Q2) + p))
    pred = np.fft.ifft2(np.fft.fft2(env) * np.exp(1j * _phi(ze, Q2))
                        * (np.exp(1j * dphi) - 1.0)) * np.exp(1j * p) / m
    _e, one, two = _one_and_two(R, 'exact', z1, z2, n, dx, w)
    meas = np.asarray(two.env) - np.asarray(one.env)
    err = float(np.max(np.abs(pred - meas))) / float(np.max(np.abs(meas)))
    assert err < 1e-3, (
        f'R={R}: the split residual is NOT the predicted kernel mismatch '
        f'({err:.3e} of it is unexplained) -- something other than the kernel '
        f'fails to compose, which WOULD be a bug')


@pytest.mark.parametrize('R', _SPLIT_RS)
def test_the_composition_residual_never_exceeds_the_correction_it_buys(R):
    """The bound that replaces the paraxial-only ``< 1e-9`` identity for the
    shipping default.  It is comparative, so it cannot erode into a
    re-baselined constant: the ambiguity introduced by WHERE a leg is split
    must stay below the non-paraxial correction the exact kernel applies on
    that same leg.  If it ever inverts, the kernel costs more than it buys and
    the default is wrong.  Measured 2026-08-06 (ratio): 0.000 at R=inf and
    703.6 m, 0.132 at -0.2, 0.554 at -0.045, 0.229 at -0.030, 0.042 at
    -0.010, 0.004 at -0.003."""
    _e, one_ex, two_ex = _one_and_two(R, 'exact')
    _e2, one_fr, _t = _one_and_two(R, 'fresnel')
    resid = _maxrel(one_ex.env, two_ex.env)
    physics = _maxrel(one_ex.env, one_fr.env)
    assert physics > 0.0, (
        'the exact and paraxial kernels are indistinguishable on this leg, so '
        'this bound has no scale -- the case has lost its discriminating power')
    assert resid < physics, (
        f'R={R}: split-vs-single residual {resid:.3e} exceeds the exact-vs-'
        f'paraxial correction {physics:.3e} on the same leg (ratio '
        f'{resid / physics:.3f})')


def _oracle_full_field(E_in, z, dx, dx_out):
    """Exact ASM of the FULL field (envelope times carrier), resampled onto the
    ``dx_out`` grid by a direct matrix inverse DFT.  Plain numpy; shares no
    code with the propagators under test.  ``ifftshift`` first so index 0 is
    the CENTRE sample, which is what makes evaluating
    ``E(x) = (1/n^2) sum S exp(i q x)`` at arbitrary x correct."""
    n = E_in.shape[0]
    q = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kz = np.sqrt(np.maximum(K * K - (q[None, :] ** 2 + q[:, None] ** 2), 0.0))
    S = np.fft.fft2(np.fft.ifftshift(E_in)) * np.exp(1j * z * kz)
    xo = (np.arange(n) - n // 2) * dx_out
    W = np.exp(1j * np.outer(xo, q)) / n
    return W @ S @ W.T


def _parab(n, d, R):
    x = (np.arange(n) - n // 2) * d
    return np.exp(1j * K * (x[None, :] ** 2 + x[:, None] ** 2) / (2.0 * R))


def test_the_oracle_reproduces_the_identity_and_the_library_asm():
    """Premise guard for the adjudication below."""
    from lumenairy.propagators.propagation import angular_spectrum_propagate
    n, dx, w, R = 256, 1e-6, 8e-6, -2e-3
    E_in = _gaussian(n, dx, w) * _parab(n, dx, R)
    same = _oracle_full_field(E_in, 0.0, dx, dx)
    assert np.linalg.norm(same - E_in) / np.linalg.norm(E_in) < 1e-12
    lib = np.asarray(angular_spectrum_propagate(E_in, 0.3e-3, WL, dx, dy=dx,
                                                bandlimit=False))
    mine = _oracle_full_field(E_in, 0.3e-3, dx, dx)
    assert np.linalg.norm(mine - lib) / np.linalg.norm(lib) < 1e-9


@pytest.mark.parametrize('R,z1,z2,w', [
    (-1.0e-3, 0.20e-3, 0.10e-3, 5.0e-6),      # m = 0.70
    (-1.0e-3, 0.10e-3, 0.05e-3, 5.0e-6),      # m = 0.85
    (+2.0e-3, 0.20e-3, 0.10e-3, 5.0e-6),      # m = 1.15, diverging
])
def test_the_exact_kernel_beats_the_paraxial_one_inside_a_scaled_step(R, z1, z2,
                                                                     w):
    """Review D10: every oracle test of the exact kernel was at ``m ~ 1``, so
    "exact" had never been adjudicated INSIDE a scaled Sziklas-Siegman step --
    where an exact kernel sits on an approximate frame and could in principle
    be WORSE than the paraxial kernel the frame was derived with.

    It is not.  Against an exact full-field ASM oracle, at envelope NA 0.083
    and m = 0.70 / 0.85 / 1.15, the exact kernel is 2.2x / 4.5x / 6.8x closer
    to the truth than the paraxial one -- and the SPLIT exact route still
    beats BOTH paraxial routes.  That is what licenses the exact default and
    bounds the composition residual as an acceptable artefact."""
    import warnings
    n, dx = 512, 0.5e-6
    env = _gaussian(n, dx, w)
    E_in = env * _parab(n, dx, R)
    m = (R + z1 + z2) / R
    truth = _oracle_full_field(E_in, z1 + z2, dx, m * dx)
    amp = np.abs(truth)
    got = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for kind in ('exact', 'fresnel'):
            one = propagate_carrier_referenced(env, R, z1 + z2, WL, dx,
                                               gap_kernel=kind)
            b1 = propagate_carrier_referenced(env, R, z1, WL, dx,
                                              gap_kernel=kind)
            two = propagate_carrier_referenced(b1.env, b1.R, z2, WL, b1.dx,
                                               gap_kernel=kind)
            got[kind] = [
                _rel(np.asarray(r.env) * _parab(n, float(r.dx), float(r.R)),
                     truth, amp) for r in (one, two)]
    assert got['exact'][0] < 0.7 * got['fresnel'][0], (
        f'the exact kernel is not decisively better inside the scaled step: '
        f'{got["exact"][0]:.3e} vs paraxial {got["fresnel"][0]:.3e}')
    assert got['exact'][1] < got['fresnel'][0], (
        f'the SPLIT exact route {got["exact"][1]:.3e} is worse than the '
        f'single-leg paraxial route {got["fresnel"][0]:.3e} -- the '
        f'composition residual would then cost more than the kernel buys')


# --------------------------------------------------------------------------
# the two legs that used to bypass the knob entirely
# --------------------------------------------------------------------------

def test_a_collimated_leg_honours_the_gap_kernel():
    """``R = +/-inf`` used to call ``fresnel_tf_propagate`` unconditionally, so
    the knob (and ``tilt``) were silently dropped on exactly the leg where the
    exact kernel is EXACT -- m = 1, no frame rescaling.  Measured before the
    fix: exact-vs-fresnel on a collimated leg was 0.000e+00."""
    env = _gaussian(256, 1e-6, 20e-6)
    ex = propagate_carrier_referenced(env, np.inf, 2e-3, WL, 1e-6,
                                      gap_kernel='exact')
    fr = propagate_carrier_referenced(env, np.inf, 2e-3, WL, 1e-6,
                                      gap_kernel='fresnel')
    au = propagate_carrier_referenced(env, np.inf, 2e-3, WL, 1e-6)
    assert not np.array_equal(np.asarray(ex.env), np.asarray(fr.env)), (
        'a collimated leg still ignores gap_kernel')
    assert np.array_equal(np.asarray(au.env), np.asarray(ex.env)), (
        "the default 'auto' does not resolve to exact on a collimated leg")
    # and it IS the validated kernel, bitwise
    ref = _exact_envelope_tf_step(env, 2e-3, WL, 1e-6, 1e-6)
    assert np.array_equal(np.asarray(ex.env), ref)
    # tilt reaches it too (it used to be dropped on this branch)
    tl = propagate_carrier_referenced(env, np.inf, 2e-3, WL, 1e-6,
                                      gap_kernel='exact', tilt=(0.1, 0.05))
    assert not np.array_equal(np.asarray(tl.env), np.asarray(ex.env))


def test_an_astigmatic_carrier_refuses_the_exact_kernel_instead_of_downgrading():
    """The exact kernel sqrt(k^2 - qx^2 - qy^2) does NOT separate, so the
    per-axis astigmatic transform cannot carry it.  It used to accept
    ``gap_kernel='exact'`` and run the paraxial per-axis kernel anyway."""
    env = _gaussian(128, 2e-6, 30e-6)
    with pytest.raises(ValueError, match='astigmatic'):
        propagate_carrier_referenced(env, (0.5, 0.7), 1e-3, WL, 2e-6,
                                     gap_kernel='exact')
    # 'auto' and 'fresnel' are accepted (documented: paraxial on this path)
    for good in ('auto', 'fresnel'):
        propagate_carrier_referenced(env, (0.5, 0.7), 1e-3, WL, 2e-6,
                                     gap_kernel=good)
