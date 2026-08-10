"""Round-2 perf items of ``docs/audits/FIX_PERF_ROUND2_2026_08_10.md``, pinned
as IDENTITIES and BOUNDS rather than as speeds.

Four changes are pinned here.

1.  ``carrier._fourier_upsample_crop`` and ``carrier._shift_envelope`` route
    their transform pair through the library's ``_fft2`` / ``_ifft2``
    dispatcher instead of RAW ``np.fft`` (single-threaded pocketfft).  This is
    the ONE class of change in this round that is not bit-identical, so it is
    pinned two ways: (a) with the dispatcher PINNED BACK to numpy the output is
    BYTE-identical to the pre-change expression -- which proves the only thing
    that changed is the backend -- and (b) with the shipped backend the
    deviation is bounded well inside 1e-14 relative with the power ratio 1.

2.  The OPL / ray-density upsamples skip their second, NaN-mask
    ``map_coordinates`` pass when the coarse array carries no NaN.  Bit-
    identical BY CONSTRUCTION (an all-zero mask makes the consuming
    ``np.where`` the identity), so the pin runs the whole element twice --
    once as shipped, once with the guard DEFEATED -- and requires
    ``tobytes()`` equality, plus that the guard actually fired (the
    ``map_coordinates`` call count drops).

3.  The ``(2, N, N)`` coordinate stack is built straight into its buffer
    instead of through ``np.indices`` + a list build.  Bit-identical.

4.  The Newton pool ships ``_spline_data`` ONCE per worker: the parent pickles
    it once and workers keep it under a CONTENT DIGEST.  Pinned on the
    protocol -- the key must move when any field of the payload moves, a
    worker must refuse a key it does not hold rather than guess, and both arg
    shapes must produce the same answer.

Nothing here measures time; the timings live in the FIX document.
"""
from __future__ import annotations

import inspect
import pickle
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT
from lumenairy.propagators import carrier as C
from lumenairy.propagators import fft_infra as FI


# ===========================================================================
# helpers
# ===========================================================================
def _smooth_env(n, seed=0, dtype=np.complex128):
    """A SMOOTH, band-limited envelope of the kind these two functions are
    only ever handed (the carrier has been divided out)."""
    rng = np.random.default_rng(seed)
    x = (np.arange(n) - n // 2) / (0.30 * n)
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    e = (np.exp(-r2) * np.exp(1j * 0.7 * r2)
         * (1.0 + 0.01 * rng.standard_normal((n, n))))
    return e.astype(dtype)


def _ref_upsample_crop(env, n_crop, n_fine):
    """The PRE-CHANGE ``_fourier_upsample_crop`` transform expression,
    verbatim: raw ``np.fft``.  Copied, not imported."""
    env = np.asarray(env)
    n = env.shape[-1]
    c0 = n // 2 - n_crop // 2
    ec = np.ascontiguousarray(env[c0:c0 + n_crop, c0:c0 + n_crop])
    if n_fine == n_crop:
        return ec
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(ec)))
    if n_fine > n_crop:
        pad = np.zeros((n_fine, n_fine), dtype=np.complex128)
        o = n_fine // 2 - n_crop // 2
        pad[o:o + n_crop, o:o + n_crop] = F
    else:
        o = n_crop // 2 - n_fine // 2
        pad = F[o:o + n_fine, o:o + n_fine]
    out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pad)))
    return out * (float(n_fine) / float(n_crop)) ** 2


def _ref_shift_envelope(env, sx, sy, dx):
    """The PRE-CHANGE ``_shift_envelope``, verbatim."""
    e = np.asarray(env)
    if sx == 0.0 and sy == 0.0:
        return e.copy()
    ny, nx = e.shape[-2], e.shape[-1]
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    ramp = np.exp(-2j * np.pi * (fx[None, :] * sx + fy[:, None] * sy))
    out = np.fft.ifft2(np.fft.fft2(e) * ramp)
    return out.astype(e.dtype, copy=False) if np.iscomplexobj(e) else out


class _NumpyFFTPinned:
    """Context manager that pins the dispatcher to ``numpy.fft`` -- the
    documented escape hatch, and the fail-before switch for items 1 / 4a."""

    def __enter__(self):
        self._pf = FI.USE_PYFFTW
        self._sf = FI.USE_SCIPY_FFT
        FI.USE_PYFFTW = False
        FI.USE_SCIPY_FFT = False
        return self

    def __exit__(self, *exc):
        FI.USE_PYFFTW = self._pf
        FI.USE_SCIPY_FFT = self._sf
        return False


def _bound(a, b):
    den = float(np.linalg.norm(a.ravel()))
    rl2 = float(np.linalg.norm((a - b).ravel()) / den) if den else 0.0
    pa = float(np.sum(np.abs(a) ** 2))
    pb = float(np.sum(np.abs(b) ** 2))
    return rl2, (pb / pa if pa else 1.0)


# ===========================================================================
# 1.  _fourier_upsample_crop through the dispatcher
# ===========================================================================
@pytest.mark.parametrize('n_crop,n_fine', [(256, 1024), (256, 128),
                                           (128, 512)])
def test_upsample_crop_byte_identical_with_the_dispatcher_pinned_to_numpy(
        n_crop, n_fine):
    """FAIL-BEFORE SWITCH.  Pin the dispatcher back to ``numpy.fft`` -- what
    this site used to call directly -- and the result must be BYTE-identical
    to the pre-change expression.  That is the proof that the edit changed the
    BACKEND and nothing else: no shift, no scale, no re-ordered spectrum."""
    e = _smooth_env(512)
    ref = _ref_upsample_crop(e, n_crop, n_fine)
    with _NumpyFFTPinned():
        got = C._fourier_upsample_crop(e, n_crop, n_fine)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert got.tobytes() == ref.tobytes()


@pytest.mark.parametrize('n_crop,n_fine', [(256, 1024), (256, 128)])
def test_upsample_crop_shipped_backend_is_bounded(n_crop, n_fine):
    """With the SHIPPED backend the two are not bit-identical (different FFT
    implementations) -- so the claim is a measured bound, not an assertion of
    equality.  1e-14 is the envelope; the campaign measured 4.4e-16 at the
    real 8192-square shapes (FIX_PERF_ROUND2 sec 2)."""
    e = _smooth_env(512)
    ref = _ref_upsample_crop(e, n_crop, n_fine)
    got = C._fourier_upsample_crop(e, n_crop, n_fine)
    rl2, pratio = _bound(ref, got)
    assert rl2 < 1e-14, f'rel L2 {rl2:.3e}'
    assert abs(pratio - 1.0) < 1e-12, f'power ratio {pratio:.15f}'


def test_upsample_crop_result_does_not_alias_a_plan_workspace():
    """``_fft2`` / ``_ifft2`` can hand back one of the plan cache's ping-pong
    workspaces.  ``fftshift`` allocates, so this function's output must be its
    own -- pinned by issuing more transforms at the same key afterwards and
    requiring the first answer to be unchanged."""
    e = _smooth_env(512)
    first = C._fourier_upsample_crop(e, 256, 1024)
    keep = first.copy()
    for _ in range(3):
        C._fourier_upsample_crop(_smooth_env(512, seed=7), 256, 1024)
    assert first.tobytes() == keep.tobytes()


def test_upsample_crop_keeps_numpys_double_only_output_dtype():
    """numpy's FFT is double-only and returned complex128 for every input
    dtype; the dispatcher's backends preserve complex64.  The promotion in the
    function is what keeps a narrow caller's output where it was."""
    for dt in (np.complex64, np.complex128):
        out = C._fourier_upsample_crop(_smooth_env(128, dtype=dt), 64, 256)
        assert out.dtype == np.complex128, dt
    # the no-transform branch is a pure crop and keeps the input dtype, as it
    # always did
    same = C._fourier_upsample_crop(_smooth_env(128, dtype=np.complex64),
                                    64, 64)
    assert same.dtype == np.complex64


def test_upsample_crop_actually_calls_the_dispatcher():
    """TEETH: the two tests above would both pass if the function had quietly
    gone back to ``np.fft`` (numpy is the pinned backend in one and a valid
    'bounded' answer in the other)."""
    seen = {'f': 0, 'i': 0}
    real_f, real_i = FI._fft2, FI._ifft2

    def _f(x):
        seen['f'] += 1
        return real_f(x)

    def _i(x):
        seen['i'] += 1
        return real_i(x)

    FI._fft2, FI._ifft2 = _f, _i
    try:
        C._fourier_upsample_crop(_smooth_env(256), 128, 512)
    finally:
        FI._fft2, FI._ifft2 = real_f, real_i
    assert seen == {'f': 1, 'i': 1}, seen


# ===========================================================================
# 4a.  _shift_envelope through the dispatcher
# ===========================================================================
def test_shift_envelope_byte_identical_with_the_dispatcher_pinned_to_numpy():
    e = _smooth_env(256)
    ref = _ref_shift_envelope(e, 1.3e-6, -0.7e-6, 1e-6)
    with _NumpyFFTPinned():
        got = C._shift_envelope(e, 1.3e-6, -0.7e-6, 1e-6)
    assert got.dtype == ref.dtype
    assert got.tobytes() == ref.tobytes()


def test_shift_envelope_shipped_backend_is_bounded_and_unaliased():
    e = _smooth_env(256)
    ref = _ref_shift_envelope(e, 1.3e-6, -0.7e-6, 1e-6)
    got = C._shift_envelope(e, 1.3e-6, -0.7e-6, 1e-6)
    rl2, pratio = _bound(ref, got)
    assert rl2 < 1e-14, f'rel L2 {rl2:.3e}'
    assert abs(pratio - 1.0) < 1e-12
    keep = got.copy()
    for _ in range(3):
        C._shift_envelope(_smooth_env(256, seed=3), 0.9e-6, 0.4e-6, 1e-6)
    assert got.tobytes() == keep.tobytes(), (
        'the returned array aliases a plan-cache workspace')


def test_shift_envelope_zero_shift_stays_a_copy_of_the_input():
    """The early return is untouched: no transform, and never the caller's
    own buffer."""
    e = _smooth_env(64)
    out = C._shift_envelope(e, 0.0, 0.0, 1e-6)
    assert out is not e and out.tobytes() == e.tobytes()


# ===========================================================================
# 3.  the coordinate stack, built straight into its buffer
# ===========================================================================
@pytest.mark.parametrize('N,sub', [(1024, 4), (512, 7),
                                   (2048, 87.14893617021276)])
def test_coords_build_is_bit_identical_to_the_indices_form(N, sub):
    ii, jj = np.indices((N, N), dtype=np.float64)
    old = np.array([ii / sub, jj / sub])
    del ii, jj
    ax = np.arange(N, dtype=np.float64) / sub
    new = np.empty((2, N, N), dtype=np.float64)
    new[0] = ax[:, None]
    new[1] = ax[None, :]
    assert np.array_equal(old, new)
    assert old.tobytes() == new.tobytes()


def test_the_element_builds_its_coords_without_np_indices():
    """TEETH for the memory half: ``np.indices`` on the wave grid is exactly
    the (2, N, N) transient the change removes, so its absence from the
    upsample block is the pin."""
    src = inspect.getsource(la.apply_real_lens_traced)
    head = src.split('_opl_up_order = ')[1]
    assert 'np.indices((N, N)' not in head, (
        'the wave-grid coordinate stack is materialising np.indices again')


# ===========================================================================
# 2.  the NaN-mask pass guard  (element-level, design-121-like)
# ===========================================================================
_WL = 1.31e-6
_N = 1024
_DX = 6.0e-6
_WBEAM = 1.2e-3
_RC = -0.06
_ZLEG = 15e-3
_AP = 9.0e-3
_RS = 4


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _presc():
    return {'name': 'perf_round2_leg', 'aperture_diameter': _AP,
            'surfaces': [_flat(), _flat()], 'thicknesses': [_ZLEG]}


def _field():
    k0 = 2.0 * np.pi / _WL
    ax = (np.arange(_N) - _N / 2) * _DX
    X, Y = np.meshgrid(ax, ax)
    rho = np.sqrt(X * X + Y * Y + _RC * _RC)
    W = -(rho - abs(_RC))
    a = 2.0 * ((X * X + Y * Y) / _WBEAM ** 2) ** 2 / k0
    amp = np.exp(-(X * X + Y * Y) / _WBEAM ** 2)
    return (amp * np.exp(1j * k0 * (W + a))).astype(np.complex128)


def _run_traced(E):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E, prescription=_presc(), wavelength=_WL, dx=_DX, carrier=_RC,
            amplitude_model='ray_density', preserve_input_phase='remap',
            remap_sampling='full', ray_subsample=_RS, parallel_amp=False,
            n_workers=1, on_undersample='silent',
            on_noncollimated='silent', fit_radius_beam_factor=2.0))


class _AlwaysAny(np.ndarray):
    """A bool array whose ``.any()`` lies.  The VALUES are untouched, so the
    NaN mask it feeds is still all-False -- only the guard's decision moves."""

    def any(self, *a, **kw):          # noqa: A003 - mirrors ndarray.any
        return True


class _ForceNaNPass:
    """Shim for ``_lens_traced.np`` that defeats the NaN-pass guard: every
    ``isnan`` result claims to contain a NaN, so the second
    ``map_coordinates`` runs exactly as it did pre-change."""

    def __init__(self, mod):
        self._mod = mod
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._mod, name)

    def isnan(self, *a, **kw):
        self.calls += 1
        return np.isnan(*a, **kw).view(_AlwaysAny)


class _CountingMapCoordinates:
    def __init__(self):
        import scipy.ndimage as snd
        self._snd = snd
        self._orig = snd.map_coordinates
        self.n = 0

    def __enter__(self):
        def _mc(inp, coords, *a, **kw):
            self.n += 1
            return self._orig(inp, coords, *a, **kw)
        self._snd.map_coordinates = _mc
        return self

    def __exit__(self, *exc):
        self._snd.map_coordinates = self._orig
        return False


@pytest.mark.slow
def test_nan_pass_guard_is_byte_identical_and_actually_fires():
    """THE pin for item 2.  One design-121-like element call: the exit field
    with the guard ACTIVE must be byte-identical to the same call with the
    guard DEFEATED (which is the pre-change behaviour), and the guard must
    have saved calls -- otherwise the identity is vacuous."""
    E = _field()
    with _CountingMapCoordinates() as cnt_on:
        got = _run_traced(E)
    shim = _ForceNaNPass(np)
    real = LT.np
    LT.np = shim
    try:
        with _CountingMapCoordinates() as cnt_off:
            ref = _run_traced(E)
    finally:
        LT.np = real
    assert shim.calls > 0, 'the shim never saw an isnan -- path changed'
    assert cnt_off.n > cnt_on.n, (
        f'the NaN-pass guard saved nothing on this fixture '
        f'({cnt_on.n} calls with it, {cnt_off.n} without) -- either the '
        f'coarse arrays carry NaN here or the guard is not wired')
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert got.tobytes() == ref.tobytes(), (
        'the exit field moved when the NaN pass was skipped; '
        f'max|delta| = {np.max(np.abs(got - ref)):.3e}')
    assert np.isfinite(np.abs(got)).all()
    assert float(np.abs(got).max()) > 0.0


def test_the_skipped_nan_pass_is_the_identity_by_construction():
    """The guard's whole argument, at the level it is made: interpolating an
    ALL-ZERO mask returns identically zero, so ``where(nan_full > 0.5, ...)``
    cannot change a bit.  Pinned at the real coarse->fine ratio design 121's
    fine retrace uses (94^2 -> a wave grid, i.e. a NON-integer ``sub``)."""
    from scipy.ndimage import map_coordinates
    Nc, N = 94, 512
    sub = N / Nc
    g = np.linspace(-1.0, 1.0, Nc)
    Xc, Yc = np.meshgrid(g, g)
    coarse = 1.7e-3 * (Xc ** 2 + Yc ** 2) + 3e-5 * Xc ** 4
    assert not np.isnan(coarse).any()
    ax = np.arange(N, dtype=np.float64) / sub
    c = np.empty((2, N, N), dtype=np.float64)
    c[0] = ax[:, None]
    c[1] = ax[None, :]
    clean = np.where(np.isnan(coarse), 0.0, coarse)
    opl = map_coordinates(clean, c, order=3, mode='nearest', prefilter=True)
    nan_full = map_coordinates(np.isnan(coarse).astype(np.float64), c,
                               order=1, mode='nearest')
    assert not nan_full.any()
    assert np.where(nan_full > 0.5, np.nan, opl).tobytes() == opl.tobytes()


def test_the_guard_still_carries_a_real_nan_mask():
    """And the other direction: with NaN present the mask must still be built
    and applied, or the guard would be a silent data loss."""
    src = inspect.getsource(la.apply_real_lens_traced)
    assert '_opl_has_nan = bool(np.isnan(opl_coarse).any())' in src
    assert 'if _opl_has_nan:' in src
    assert '_rd_has_nan = bool(np.isnan(ard_coarse).any())' in src


# ===========================================================================
# 4b.  _poly elides the exponent-0 / exponent-1 power-table entries
# ===========================================================================
_DEG = 6
_TERMS = [(i, d - i) for d in range(1, _DEG + 1) for i in range(d + 1)]


def _poly_no_elision(u, v, coef, terms, hess):
    """The PRE-ELISION accumulation loop, verbatim (the power-cached form that
    shipped on 2026-08-09).  Copied, not imported."""
    need_u, need_v = set(), set()
    for c, (i, j) in zip(coef, terms):
        if c == 0.0:
            continue
        need_u.add(i)
        need_v.add(j)
        if i >= 1:
            need_u.add(i - 1)
        if j >= 1:
            need_v.add(j - 1)
        if hess:
            if i >= 2:
                need_u.add(i - 2)
            if j >= 2:
                need_v.add(j - 2)
    UP = {p: u ** p for p in sorted(need_u)}
    VP = {q: v ** q for q in sorted(need_v)}
    dt = np.result_type(u, coef) if len(terms) else np.result_type(u)
    P = np.zeros_like(u, dtype=dt)
    Pu = np.zeros_like(u, dtype=dt)
    Pv = np.zeros_like(u, dtype=dt)
    Puu = Puv = Pvv = None
    if hess:
        Puu = np.zeros_like(u, dtype=dt)
        Puv = np.zeros_like(u, dtype=dt)
        Pvv = np.zeros_like(u, dtype=dt)
    for c, (i, j) in zip(coef, terms):
        if c == 0.0:
            continue
        P += c * UP[i] * VP[j]
        if i >= 1:
            Pu += c * i * UP[i - 1] * VP[j]
        if j >= 1:
            Pv += c * j * UP[i] * VP[j - 1]
        if hess:
            if i >= 2:
                Puu += c * i * (i - 1) * UP[i - 2] * VP[j]
            if i >= 1 and j >= 1:
                Puv += c * i * j * UP[i - 1] * VP[j - 1]
            if j >= 2:
                Pvv += c * j * (j - 1) * UP[i] * VP[j - 2]
    return P, Pu, Pv, Puu, Puv, Pvv


@pytest.mark.parametrize('hess', [False, True])
def test_poly_exponent_elision_is_byte_identical(hess):
    """Multiplying by ``u ** 0`` (an array of ones) is exact, and ``u ** 1``
    is ``u``'s bits -- so eliding both cannot move a result.  Asserted against
    a verbatim copy of the pre-elision loop, on the real design-121 term list
    at the shipped degree, over a 2-D band and a scalar query."""
    rng = np.random.default_rng(4)
    coef = rng.normal(size=len(_TERMS)) * 1e-6
    s = 3.0e-3
    for shape in ((97, 131), ()):
        # Drive BOTH arms from the same ``ex``/``ey`` and let each derive
        # ``u = ex / s`` itself, exactly as ``_poly`` does.  Handing the
        # reference a ``u`` and the method a ``u * s`` compares two different
        # inputs -- ``(u * s) / s`` is not ``u`` for ~15 % of random float64s,
        # and an earlier revision of this test did precisely that (it passed
        # on Windows and failed on WSL/numpy 2.4.6 at 4.3e-19, which is the
        # round-trip, not the elision).
        ex = rng.normal(size=shape) * 0.8 * s if shape else np.float64(0.37 * s)
        ey = rng.normal(size=shape) * 0.8 * s if shape else np.float64(-0.21 * s)
        u = ex / s
        v = ey / s
        ek = _ResidualEikonal_for(coef, s)
        got = ek._poly(ex, ey, hess=hess)
        ref = _poly_no_elision(u, v, coef, _TERMS, hess)
        for k, (g, r) in enumerate(zip(got, ref)):
            if k == 0:
                exp = r
            elif k in (1, 2):
                exp = None if r is None else r / s
            else:
                exp = None if r is None else r / (s * s)
            if exp is None:
                assert g is None, k
                continue
            assert np.asarray(g).tobytes() == np.asarray(exp).tobytes(), (
                f'slot {k} moved: max|d| '
                f'{np.max(np.abs(np.asarray(g) - np.asarray(exp))):.3e}')


def _ResidualEikonal_for(coef, scale):
    """A ``_ResidualEikonal`` whose ``_poly`` sees exactly ``coef`` / _TERMS."""
    return LT._ResidualEikonal(np.asarray(coef, dtype=np.float64), _TERMS,
                               0.0, 0.0, float(scale), np.inf)


def test_poly_never_materialises_an_exponent_zero_or_one_power():
    """TEETH: the identity above would pass on a function that quietly went
    back to building the whole table."""
    src = inspect.getsource(LT._ResidualEikonal._poly)
    assert 'for p in sorted(need_u) if p}' in src, (
        'the exponent-0 entry is being built again'
    )
    assert '(u if p == 1 else u ** p)' in src, (
        'the exponent-1 entry is being copied again')
    assert 'def _mul(' in src and '_mul(c, i, j)' in src


# ===========================================================================
# 4.  Newton pool payload residency
# ===========================================================================
def _payload(fit_edge=48, order=6):
    g = np.linspace(-9.2e-3, 9.2e-3, fit_edge)
    Xf, Yf = np.meshgrid(g, g)
    n_c = (order + 1) * (order + 2) // 2
    return {
        'xs_in': g, 'x_out_grid': Xf * 0.98, 'y_out_grid': Yf * 0.98,
        'opl_grid': (Xf ** 2 + Yf ** 2) * 1.0e2,
        'launch_radius': 9.2e-3, 'dx': 1e-6, 'bound': 9.2e-3,
        'inv_M_x': 1.1, 'inv_M_y': 1.1,
        'newton_fit': 'polynomial', 'newton_max_iters': 12,
        'cheb_backend': 'numpy',
        'cheb_fit': {'x_out': {'coeffs': np.zeros(n_c)}},
    }


def test_the_payload_key_is_content_derived():
    """The whole safety argument for reusing a resident payload.  The dict is
    REBUILT per call but MUTATED per dispatch (``cheb_backend`` /
    ``cheb_fit``), so an identity- or counter-keyed cache could serve a stale
    backend pin -- a silently different floating-point order.  Every field
    must move the key."""
    p = _payload()
    k0, b0 = LT._newton_payload_blob(p)
    assert LT._newton_payload_blob(p)[0] == k0, 'key is not stable'
    for field, value in (('cheb_backend', 'numba'),
                         ('newton_max_iters', 13),
                         ('newton_fit', 'spline'),
                         ('dx', 2e-6)):
        q = dict(p)
        q[field] = value
        assert LT._newton_payload_blob(q)[0] != k0, field
    q = dict(p)
    q['cheb_fit'] = {'x_out': {'coeffs': np.ones(len(
        p['cheb_fit']['x_out']['coeffs']))}}
    assert LT._newton_payload_blob(q)[0] != k0, 'cheb_fit'
    q = dict(p)
    q['opl_grid'] = p['opl_grid'] * 1.0000001
    assert LT._newton_payload_blob(q)[0] != k0, 'opl_grid'
    assert pickle.loads(b0)['cheb_backend'] == 'numpy'


def test_a_worker_refuses_a_key_it_does_not_hold():
    """Residency is an optimisation, never a promise: a worker asked for
    bytes it never received must say so, so the parent re-sends -- not guess
    from whatever it happens to hold."""
    LT._newton_pool_init()
    p = _payload()
    key, blob = LT._newton_payload_blob(p)
    with pytest.raises(LT.NewtonPayloadNotResident):
        LT._newton_worker_payload(key, None)
    got = LT._newton_worker_payload(key, blob)
    assert got['cheb_backend'] == p['cheb_backend']
    assert LT._newton_worker_payload(key, None) is got
    # a SECOND payload evicts the first: at most one resident per worker
    p2 = dict(p)
    p2['cheb_backend'] = 'numba'
    key2, blob2 = LT._newton_payload_blob(p2)
    LT._newton_worker_payload(key2, blob2)
    assert len(LT._WORKER_PAYLOADS) == 1
    with pytest.raises(LT.NewtonPayloadNotResident):
        LT._newton_worker_payload(key, None)
    LT._newton_pool_init()
    assert LT._WORKER_PAYLOADS == {}


def test_the_chunk_worker_accepts_both_arg_shapes_identically():
    """The historical 3-tuple is still served (a direct caller, an old pinned
    payload), and the 4-tuple must give the SAME answer -- bit for bit."""
    LT._newton_pool_init()
    p = _payload(fit_edge=40)
    ev = LT._Cheb2DEvaluator(p['xs_in'], p['xs_in'], p['opl_grid'], order=6,
                             xp=np)
    p['cheb_fit'] = LT._cheb_fit_payload(ev, ev, ev, 'polynomial')
    p['cheb_backend'] = 'numpy'
    xs = np.linspace(-5e-3, 5e-3, 97)
    ys = xs[::-1].copy()
    old, n_old = LT._newton_invert_chunk((p, xs.copy(), ys.copy()))
    key, blob = LT._newton_payload_blob(p)
    new, n_new = LT._newton_invert_chunk((key, blob, xs.copy(), ys.copy()))
    res, n_res = LT._newton_invert_chunk((key, None, xs.copy(), ys.copy()))
    assert n_old == n_new == n_res
    assert old.tobytes() == new.tobytes() == res.tobytes()
    LT._newton_pool_init()


def test_the_pool_is_built_with_the_payload_initializer_and_close_resets():
    """Wiring pins.  The initializer is what gives a worker a known-empty
    registry, and ``close_worker_pool`` must drop the parent's residency
    belief -- the workers it was about are gone."""
    src = inspect.getsource(LT._get_persistent_worker_pool)
    assert 'initializer=_newton_pool_init' in src
    assert '_POOL_RESIDENT_PAYLOAD_KEY = None' in src
    csrc = inspect.getsource(LT.close_worker_pool)
    assert '_POOL_RESIDENT_PAYLOAD_KEY = None' in csrc
    LT._POOL_RESIDENT_PAYLOAD_KEY = 'stale'
    LT.close_worker_pool()
    assert LT._POOL_RESIDENT_PAYLOAD_KEY is None


def test_the_dispatch_sends_the_blob_once_and_the_key_thereafter():
    """The mechanism, read off the dispatch closure: one
    ``_newton_payload_blob`` per dispatch, the blob only when the belief does
    not match, and a miss re-submitted WITH the payload."""
    src = inspect.getsource(la.apply_real_lens_traced)
    body = src[src.index('def _invert_newton_parallel'):]
    body = body[:body.index('def _support_taper')]
    assert body.count('_newton_payload_blob(_spline_data)') == 1
    assert ('_send = (None if _POOL_RESIDENT_PAYLOAD_KEY == _pkey else _pblob)'
            in body)
    assert 'except NewtonPayloadNotResident' in body
    assert body.index('except NewtonPayloadNotResident') < body.index(
        'except (BrokenProcessPool'), (
        'the pool-infrastructure clause now shadows the residency miss, so a '
        'miss falls all the way back to serial instead of re-sending')
    assert '(_pkey, _pblob, _chunks[i][0]' in body, (
        'the re-submit no longer attaches the payload')
