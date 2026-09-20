"""Wave-5 hygiene-2, H2-2 (audit item 18) -- ``_collins_transport`` on JAX.

Before 5.48.0 the Collins transport was NumPy-only, not by design but by
plumbing: it called ``np.asarray`` / ``np.ascontiguousarray(..., dtype=
np.complex128)`` and then ``_fft2``, ``_collins_angle_support``,
``_collins_exact_kernel_correction``, ``_collins_space_support``,
``_collins_sampling_stats`` and ``_bluestein_centred_2d``, and the only backend
any of that could reach was NumPy's.  The fix is ONE ``xp`` threaded through
the chain -- no ``_jax`` twin of anything -- using the module's own
``(xp, is_jax, bld)`` triple (:func:`~lumenairy.propagators.carrier.
_backend_of`), its own device-move helper (``_to_dev``), its own
``exp(i*phase)`` builder (``_tf_phase_to_H``, already shared by
``_exact_tf_2d_xp`` and ``_fresnel_tf_2d_xp``) and one new FFT-pair selector
(``_fft2_pair``) that IS ``lumenairy.backend.fft2``'s dispatch.

THE BARS, AND WHERE THEY COME FROM.

Nothing here compares JAX to NumPy against a number somebody liked.  The bar is
MEASURED FIRST, in :func:`_fft_spread_bar`, from the only thing that can
legitimately differ: the two backends' FFTs.  NumPy's path is pyFFTW or
scipy.fft's pocketfft; JAX's is XLA's own transform.  They are different
algorithms over the same arithmetic, so they differ in the last bits, and the
transport applies two of them (the measurement FFT and the chirp-Z's pair) plus
the Bluestein convolution's three.  The bar is the measured single-transform
spread times a stated factor for the chain depth, re-measured on the running
build at the FIXTURE's own shape -- so it tracks a numpy release, an XLA
release, or a change of FFT backend rather than pinning today's.

``jax_enable_x64`` is forced on: in float32 the comparison would be measuring
JAX's default dtype policy, not the port.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy.propagators.carrier as CA

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

WL = 633e-9
N = 64
DX = 8e-6
R_IN = -0.05
Z = 5e-3
R_REF = -0.045


@pytest.fixture(scope="module", autouse=True)
def _x64():
    """float64 everywhere.  A float32 JAX would make every bar below a
    statement about dtype policy instead of about the port."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _gauss(n=N, dx=DX, w=60e-6, dtype=np.complex128):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(dtype)


def _rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _fft_spread_bar(env, chain_depth=6.0):
    """The bar, MEASURED on this build, this shape, this pair of FFTs.

    One forward transform of the fixture through each backend's own 2-D FFT,
    compared relatively.  That is the irreducible per-transform disagreement.
    ``chain_depth`` is an UPPER BOUND on how many the transport applies: the
    measurement transform (1), the exact-kernel correction's inverse (1, on the
    'exact' arm only) and the Bluestein primitive's three (two 2-D FFTs plus
    the chirp-kernel transform) -- five on the 2-D arm, and six on the
    separable arm, which spends three 1-D passes per axis instead.  Round-off
    through a chain of unitary transforms adds at worst linearly, so
    ``chain_depth`` times the single-transform spread is a bound and not a fit.

    A floor of ``32 * eps`` is added because a spread measured as exactly zero
    (the two backends agreeing bit for bit on a small easy case, which does
    happen) would otherwise make the bar zero and the test a coin toss.
    """
    from lumenairy.propagators.fft_infra import _fft2
    a = np.asarray(_fft2(np.ascontiguousarray(env, dtype=np.complex128)))
    b = np.asarray(jnp.fft.fft2(jnp.asarray(env, dtype=jnp.complex128)))
    spread = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    return max(chain_depth * spread, 32.0 * float(np.finfo(np.float64).eps))


# ===========================================================================
# 1. The chain really is ONE implementation
# ===========================================================================

def test_the_fft_pair_is_the_librarys_own_dispatcher_not_a_second_copy():
    """``_fft2_pair`` must hand back the library's existing callables, not
    wrap them.

    Two claims.  (a) On the NumPy side the callables are ``fft_infra._fft2`` /
    ``_ifft2`` BY IDENTITY -- and identity is the point, not equivalence:
    ``_bluestein_2d`` keys its chirp-kernel FFT cache on
    ``fft2 is fft_infra._fft2``, so a wrapper would silently disable that cache
    for every Collins leg.  (b) On both sides the callable agrees with
    ``lumenairy.backend.fft2``, which is the library's declared backend FFT
    entry point, so there is one dispatch and not two.
    """
    from lumenairy.backend import fft2 as backend_fft2
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    f, i = CA._fft2_pair(np, False)
    assert f is _fft2 and i is _ifft2

    env = _gauss()
    assert np.array_equal(np.ascontiguousarray(f(np.ascontiguousarray(
        env, dtype=np.complex128))).view(np.float64),
        np.ascontiguousarray(backend_fft2(np.ascontiguousarray(
            env, dtype=np.complex128))).view(np.float64))

    fj, ij = CA._fft2_pair(jnp, True)
    ej = jnp.asarray(env, dtype=jnp.complex128)
    assert np.array_equal(np.asarray(fj(ej)), np.asarray(backend_fft2(ej)))
    assert np.array_equal(np.asarray(ij(ej)),
                          np.asarray(jnp.fft.ifft2(ej)))


def test_no_jax_twin_of_the_collins_chain_exists():
    """The rule this package is held to: ONE ``xp``-parametrised
    implementation per kernel, no per-flavour copies.

    A structural inventory, so a future ``_collins_transport_jax`` is caught by
    the gate rather than by a reviewer.  Every private Collins name in the
    module is listed and none may carry a backend suffix; and the module must
    not import a ``*_jax*`` sibling for this chain.

    The source half of the check reads the module's CODE, docstrings stripped
    (2026-09-19).  A raw substring search over the whole file fired on a
    docstring that merely CITED this test file by name -- a false positive on
    a cross-reference, which is the opposite of what the gate is for.  Stripped
    of docstrings the search still covers every import, every definition and
    every call, which is where a per-flavour twin would have to appear.
    """
    import ast
    import pathlib
    names = [n for n in dir(CA) if n.startswith('_collins')]
    assert names, "the Collins chain vanished; this inventory is stale"
    for n in names:
        low = n.lower()
        for suffix in ('_jax', '_jnp', '_cupy', '_cp', '_np', '_numpy'):
            assert not low.endswith(suffix), (
                f"{n} looks like a per-backend copy of a Collins helper; the "
                f"chain is meant to be one xp-parametrised implementation")
    src = pathlib.Path(CA.__file__).read_text(encoding='cp1252')
    tree = ast.parse(src)
    code = '\n'.join(
        _body_source(src, n) for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef))
    # module-level statements too, minus the module docstring
    code += '\n' + '\n'.join(
        ast.get_source_segment(src, s) or ''
        for s in tree.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                and isinstance(s.value.value, str))
        and not isinstance(s, ast.FunctionDef))
    assert '_collins_jax' not in code and 'collins_jax' not in code


def _body_source(src, node):
    """A function's source with its DOCSTRING removed.

    A census that reads docstrings counts a cross-reference as a second
    implementation.  Every statement is taken through
    ``ast.get_source_segment`` except a leading string expression.
    """
    import ast
    body = node.body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    return '\n'.join(ast.get_source_segment(src, s) or '' for s in body)


def test_the_exact_dispersion_is_written_once():
    """ONE kernel, ONE implementation -- the campaign's standing rule, gated.

    MEASURED 2026-09-19 (VERIFY-WAVE5-HYGIENE2 V-D22): THREE transcriptions of
    ``sqrt(k^2 - |k s + q|^2) - k N + (s.q)/N`` lived in ``carrier.py`` --
    ``_exact_tf_2d_xp`` (``xp``-parametrised), ``_exact_envelope_tf_step``
    (NumPy-only, and written twice inside itself), and
    ``_collins_exact_kernel_correction`` (``(xp, is_jax, bld)``-parametrised
    by H2-2) -- with the same ``|s|^2 < 1`` guard written three times under
    three different error prefixes.  They are one kernel with two uses: the
    first two add the piston ``k z``, the third subtracts the paraxial
    ``|q|^2/(2k)`` instead.

    Three independent tokens are censused, not one, because a future author
    could split any one of them off on its own: the ``q = 0`` subtraction
    (``root0``), the radical's shifted-frequency form, and the evanescent
    guard.  Docstrings are stripped, so a cross-reference is not counted as an
    implementation.

    This is the gate that had to LAND WITH the consolidation.  The only thing
    that caught a conjugated exact kernel before it was
    ``test_audit2609_b4_collins_transport.py::TestSameTheorem``, an AGREEMENT
    test between two of these transcriptions -- and MEASURED on the
    consolidated tree, a sign flip of the single kernel leaves that file
    reading 9 passed, because both transports now move together.  The sign is
    pinned directly instead, in
    ``test_wave5_h2_near_focus_table.py::
    test_the_measured_departure_is_the_quartic_times_one_constant``.
    """
    import ast
    import pathlib
    src = pathlib.Path(CA.__file__).read_text(encoding='cp1252')
    tree = ast.parse(src)
    defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert len(defs) > 100, (
        f"only {len(defs)} function definitions parsed out of carrier.py; the "
        f"census is not reading the module")

    def owners(token):
        return sorted({n.name for n in defs
                       if token in _body_source(src, n)})

    for token, what in (
            ('root0', 'the q = 0 subtraction k N'),
            ('ax * ax + ay * ay', 'the shifted-frequency radical'),
            ('s2 < 1.0', 'the evanescent-carrier guard')):
        got = owners(token)
        assert got == ['_exact_dispersion_phase'], (
            f"{what} ({token!r}) is implemented in {got} -- it must live in "
            f"_exact_dispersion_phase and nowhere else.  ONE "
            f"xp-parametrised implementation per numerical kernel is this "
            f"campaign's standing rule, and the specific cost of breaking it "
            f"here is measured: with two copies, the only guard on the "
            f"kernel's SIGN was an agreement test between them.")

    # ... and the one implementation is reached from all three sites.
    for site in ('_exact_tf_2d_xp', '_exact_envelope_tf_step',
                 '_collins_exact_kernel_correction'):
        node = next(n for n in defs if n.name == site)
        assert '_exact_dispersion_phase(' in _body_source(src, node), (
            f"{site} no longer calls _exact_dispersion_phase; a fourth "
            f"transcription has appeared, or the site was deleted")


@pytest.mark.parametrize("name", ('_collins_exact_kernel_correction',
                                  '_collins_axis_chirp'))
def test_the_threaded_helpers_default_to_the_numpy_build(name):
    """Every helper that gained a backend argument defaults to the historical
    NumPy build, which is why every existing caller and every existing test
    kept working without being touched."""
    import inspect
    sig = inspect.signature(getattr(CA, name))
    if name == '_collins_axis_chirp':
        assert sig.parameters['bld'].default is np
    else:
        assert sig.parameters['xp'].default is np
        assert sig.parameters['is_jax'].default is False
        assert sig.parameters['bld'].default is np


# ===========================================================================
# 2. The JAX path agrees with NumPy to the FFTs' own spread
# ===========================================================================

@pytest.mark.parametrize("gap_kernel", ('auto', 'fresnel', 'exact'))
def test_the_private_transport_agrees_across_backends(gap_kernel):
    """``_collins_transport`` itself, both backends, both kernels.

    The bar is measured on this build first (:func:`_fft_spread_bar`), and the
    assertion carries BOTH sides of it: the disagreement must be below the
    bar, and the bar must be below the smallest real signal -- which here is
    the exact-kernel refinement's own departure from the paraxial kernel,
    measured in the same call.  If the two ever met, the test would be reading
    noise and says so instead of passing.
    """
    env = _gauss()
    bar = _fft_spread_bar(env)
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              gap_kernel=gap_kernel, on_collins_sampling='ignore')
    a = np.asarray(CA._collins_transport(env, R_IN, Z, WL, DX, DX, **kw))
    b = np.asarray(CA._collins_transport(
        jnp.asarray(env, dtype=jnp.complex128), R_IN, Z, WL, DX, DX, **kw))
    got = _rel(b, a)
    assert got < bar, (
        f"JAX vs NumPy {got:.3e} exceeds the measured FFT-spread bar "
        f"{bar:.3e} at gap_kernel={gap_kernel!r}")
    # the two-sided half: the bar has to be below something real
    fres = np.asarray(CA._collins_transport(
        env, R_IN, Z, WL, DX, DX, **dict(kw, gap_kernel='fresnel')))
    exact = np.asarray(CA._collins_transport(
        env, R_IN, Z, WL, DX, DX, **dict(kw, gap_kernel='exact')))
    signal = _rel(exact, fres)
    assert bar < signal / 10.0, (
        f"the measured bar {bar:.3e} is not a decade below the smallest real "
        f"signal on this fixture ({signal:.3e} between the exact and paraxial "
        f"kernels) -- the comparison is testing noise")


@pytest.mark.parametrize("transport", ('collins', 'sziklas'))
def test_the_public_entry_agrees_across_backends(transport):
    """The public entry point, so the claim is about what a caller gets."""
    env = _gauss()
    bar = _fft_spread_bar(env)
    kw = dict(wavelength=WL, dx=DX, transport=transport,
              gap_kernel='fresnel')
    if transport == 'collins':
        kw['on_collins_sampling'] = 'ignore'
    a = CA.propagate_carrier_referenced(env, R_IN, Z, **kw)
    b = CA.propagate_carrier_referenced(
        jnp.asarray(env, dtype=jnp.complex128), R_IN, Z, **kw)
    assert _rel(b.env, a.env) < bar
    assert float(b.R) == float(a.R)
    assert float(b.dx) == float(a.dx)


def test_the_astigmatic_arm_crosses_backends_too():
    """The per-axis arithmetic, which is a different code path in the same
    function (two ``A``, two ``D``, two chirps)."""
    env = _gauss()
    bar = _fft_spread_bar(env)
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
              R_ref=(-0.045, -0.07), gap_kernel='fresnel',
              on_collins_sampling='ignore')
    a = np.asarray(CA._collins_transport(env, (-0.05, -0.08), Z, WL, DX, DX,
                                         **kw))
    b = np.asarray(CA._collins_transport(
        jnp.asarray(env, dtype=jnp.complex128), (-0.05, -0.08), Z, WL, DX,
        DX, **kw))
    assert _rel(b, a) < bar


def test_a_tilted_exact_kernel_crosses_backends():
    """The tilt expansion is the one branch inside the exact-kernel correction
    that only runs when ``(L, M) != 0``, so it needs its own arm."""
    env = _gauss()
    bar = _fft_spread_bar(env)
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              gap_kernel='exact', tilt=(0.12, 0.03),
              on_collins_sampling='ignore')
    a = np.asarray(CA._collins_transport(env, R_IN, Z, WL, DX, DX, **kw))
    b = np.asarray(CA._collins_transport(
        jnp.asarray(env, dtype=jnp.complex128), R_IN, Z, WL, DX, DX, **kw))
    assert _rel(b, a) < bar


def test_the_backend_does_not_silently_upcast_a_complex64_envelope():
    """The v4.14.1 no-silent-upcast contract, on the ported path."""
    env = _gauss(dtype=np.complex64)
    kw = dict(dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
              gap_kernel='fresnel', on_collins_sampling='ignore')
    assert CA._collins_transport(env, R_IN, Z, WL, DX, DX,
                                 **kw).dtype == np.complex64
    out = CA._collins_transport(jnp.asarray(env, dtype=jnp.complex64), R_IN,
                                Z, WL, DX, DX, **kw)
    assert out.dtype == jnp.complex64


def test_the_numpy_path_is_untouched_by_the_backend_plumbing():
    """The regression that matters most: the NumPy answer must be what it was.

    Byte equality against the chain spelled out by hand -- pre-chirp, centred
    chirp-Z, post-chirp, prefactor -- using the module's own helpers with
    their NumPy defaults.  If the plumbing ever changed a NumPy expression,
    this hand-spelling and the function would part company.

    (The archive-to-archive proof against 5.47.0 is
    ``validation/probe_wave5_hyg2/colbit_{win,wsl}_compare.json``: 84 keys,
    identical on both builds.  This is its in-suite counterpart.)
    """
    from lumenairy.propagators._bluestein import _bluestein_centred_2d
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    env = _gauss()
    Ax, B, Cx, Dx = CA._collins_envelope_abcd(R_IN, Z, R_REF)
    cdt = env.dtype
    k = 2.0 * np.pi / WL
    g = env * CA._collins_axis_chirp(N, DX, WL, B / Ax, dtype=cdt)[None, :]
    g = g * CA._collins_axis_chirp(N, DX, WL, B / Ax, dtype=cdt)[:, None]
    alpha = float(DX) * float(DX) / (WL * B)
    G = _bluestein_centred_2d(
        np.ascontiguousarray(g, dtype=cdt), alpha, alpha, N, N,
        k_centre_out_x=N / 2.0, k_centre_out_y=N / 2.0,
        sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2, target_cdtype=cdt,
        separable=bool(CA._EXACT_READOUT_SEPARABLE_BLUESTEIN))
    G = G * CA._collins_axis_chirp(N, DX, WL, B / Dx, offset=-0.0,
                                   dtype=cdt)[None, :]
    G = G * CA._collins_axis_chirp(N, DX, WL, B / Dx, offset=-0.0,
                                   dtype=cdt)[:, None]
    want = complex(np.exp(1j * k * B) * DX * DX / (1j * WL * B)) * G

    got = CA._collins_transport(
        env, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N, N_out_y=N,
        R_ref=R_REF, gap_kernel='fresnel', on_collins_sampling='ignore')
    assert np.array_equal(np.ascontiguousarray(got).view(np.float64),
                          np.ascontiguousarray(want).view(np.float64))


# ===========================================================================
# 3. Under a trace: the two measured decisions are refused, not defaulted
# ===========================================================================

def _merit_factory(**kw):
    def merit(amp):
        env = amp.astype(jnp.complex128)
        out = CA._collins_transport(env, R_IN, Z, WL, DX, DX,
                                    dx_out=DX, dy_out=DX, N_out_x=N,
                                    N_out_y=N, R_ref=R_REF, **kw)
        return jnp.sum(jnp.abs(out) ** 2)
    return merit


@pytest.mark.parametrize("kw,needle", (
    (dict(gap_kernel='auto', on_collins_sampling='ignore'), "gap_kernel"),
    (dict(gap_kernel='fresnel', on_collins_sampling='warn'),
     "on_collins_sampling"),
    (dict(gap_kernel='auto', on_collins_sampling='warn'), "gap_kernel"),
))
def test_a_traced_envelope_refuses_the_decisions_it_cannot_measure(kw, needle):
    """A Tracer has no entries, so the ``k4`` kernel resolution and the Kelly
    guard cannot be evaluated.  Both are REFUSED, loudly, with the spelling
    that takes each decision named in the message.

    Not defaulted: silently taking 'fresnel' under a trace would be the same
    shape as the ``gap_kernel`` typo-falls-through defect (D4) that the
    vocabulary gate exists to remove, and a conservative grid-edge guard would
    refuse legs whose measured departure from the analytic ABCD field is
    5.6e-08 of peak.
    """
    amp = jnp.asarray(np.real(_gauss()))
    with pytest.raises(ValueError, match="Tracer") as exc:
        jax.grad(_merit_factory(**kw))(amp)
    assert needle in str(exc.value)
    assert "trace-safe" in str(exc.value)


def test_stats_out_is_refused_under_a_trace():
    """``stats_out`` asks for readings that do not exist for a traced array."""
    amp = jnp.asarray(np.real(_gauss()))

    def merit(a):
        out = CA._collins_transport(
            a.astype(jnp.complex128), R_IN, Z, WL, DX, DX, dx_out=DX,
            dy_out=DX, N_out_x=N, N_out_y=N, R_ref=R_REF,
            gap_kernel='fresnel', on_collins_sampling='ignore',
            stats_out={})
        return jnp.sum(jnp.abs(out) ** 2)

    with pytest.raises(ValueError, match="stats_out"):
        jax.grad(merit)(amp)


def test_an_eager_jax_array_is_not_a_tracer_and_measures_normally():
    """The refusal must be about TRACING, not about JAX.  An eager JAX array
    has concrete entries, so both decisions are taken exactly as on NumPy --
    including the warning the guard raises."""
    env = jnp.asarray(_gauss(w=500e-6), dtype=jnp.complex128)
    st = {}
    out = CA._collins_transport(env, -0.30, 0.29, 1.55e-6, 40e-6, 40e-6,
                                dx_out=40e-6 * 16, dy_out=40e-6 * 16,
                                N_out_x=N, N_out_y=N, R_ref=float('inf'),
                                gap_kernel='auto',
                                on_collins_sampling='ignore', stats_out=st)
    assert out.shape == (N, N)
    assert set(('k1', 'k2', 'k3', 'k4', 'kernel')) <= set(st)
    assert st['kernel'] in ('exact', 'fresnel')
    with pytest.warns(RuntimeWarning, match="under-sampled"):
        CA._collins_transport(env, -0.30, 0.29, 1.55e-6, 40e-6, 40e-6,
                              dx_out=40e-6 * 16, dy_out=40e-6 * 16,
                              N_out_x=N, N_out_y=N, R_ref=float('inf'),
                              gap_kernel='auto', on_collins_sampling='warn')


# ===========================================================================
# 4. The gradient, against a central finite difference (a ladder)
# ===========================================================================

def test_jax_grad_through_the_collins_transport_matches_a_central_difference():
    """``jax.grad`` of a scalar merit through the transport, against an
    independent central finite difference -- on a LADDER of step sizes, with
    the bar derived from the difference's own error rather than chosen.

    A central difference of a merit ``P`` has error ``~ P''' h^2 / 6``
    (truncation) plus ``~ eps |P| / h`` (cancellation), so its accuracy is a
    U-curve in ``h`` and no single ``h`` is "the" answer.  The ladder is
    scanned and the claim is made where the two terms are balanced: the BEST
    agreement over the ladder must fall below the smallest achievable
    difference error, ``~ (eps |P|)^(2/3)`` in relative terms, times a stated
    factor of ten.  Hard-failing only when the whole ladder is exhausted is
    the standard this repository holds fail-before demonstrations to, and the
    same reasoning applies to a gradient check.

    The merit is transported POWER, which the leg conserves to grid accuracy,
    so the gradient also has an ANALYTIC form: ``d/da_ij sum|u_out|^2`` is
    proportional to ``a_ij``.  That is asserted separately as a shape check
    (correlation), because the proportionality constant carries the leg's own
    area scaling.
    """
    amp0 = jnp.asarray(np.real(_gauss()))
    merit = _merit_factory(gap_kernel='fresnel',
                           on_collins_sampling='ignore')
    g = np.asarray(jax.grad(merit)(amp0))
    assert np.all(np.isfinite(g))

    a = np.asarray(amp0)
    ij = np.unravel_index(int(np.argmax(a)), a.shape)
    P0 = float(merit(amp0))
    eps = float(np.finfo(np.float64).eps)
    # the best a central difference can do, relatively: balancing eps|P|/h
    # against P''' h^2/6 puts the optimum at h ~ (eps)^(1/3) with a relative
    # error ~ eps^(2/3).
    floor = eps ** (2.0 / 3.0)
    bar = 10.0 * floor

    best = np.inf
    best_h = None
    for h in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5):
        def _at(sign):
            ap = a.copy()
            ap[ij] += sign * h
            return float(merit(jnp.asarray(ap)))
        fd = (_at(+1) - _at(-1)) / (2.0 * h)
        rel = abs(fd - g[ij]) / abs(g[ij])
        if rel < best:
            best, best_h = rel, h
    assert best < bar, (
        f"the best central difference over the ladder disagrees with jax.grad "
        f"by {best:.3e} (at h={best_h}), past {bar:.3e} = 10 x the "
        f"difference's own eps^(2/3) floor; P0={P0:.6e}")

    # the analytic SHAPE of the gradient, where the beam has support
    m = a > 0.05 * a.max()
    corr = float(np.corrcoef(g[m], a[m])[0, 1])
    assert corr > 1.0 - 1e-6, (
        f"the gradient of transported power is not proportional to the input "
        f"amplitude (correlation {corr:.9f}); the leg is not conserving power "
        f"or the gradient is not the one it should be")


def test_the_gradient_is_not_trivially_zero_or_constant():
    """The falsification arm: a gradient that were all-zero, or all-equal,
    would pass a correlation test by accident on a degenerate fixture."""
    amp0 = jnp.asarray(np.real(_gauss()))
    g = np.asarray(jax.grad(_merit_factory(
        gap_kernel='fresnel', on_collins_sampling='ignore'))(amp0))
    assert float(np.max(np.abs(g))) > 0.0
    assert float(np.std(g)) / float(np.max(np.abs(g))) > 1e-3


def test_jit_of_the_transport_matches_the_eager_traced_call():
    """``jax.jit`` traces the same way ``jax.grad`` does, so the refusal and
    the allowed spelling must behave identically under it."""
    env = jnp.asarray(_gauss(), dtype=jnp.complex128)

    def f(e):
        return CA._collins_transport(
            e, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N,
            N_out_y=N, R_ref=R_REF, gap_kernel='fresnel',
            on_collins_sampling='ignore')

    eager = np.asarray(f(env))
    jitted = np.asarray(jax.jit(f)(env))
    bar = _fft_spread_bar(np.asarray(env))
    assert _rel(jitted, eager) < bar

    def bad(e):
        return CA._collins_transport(
            e, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX, N_out_x=N,
            N_out_y=N, R_ref=R_REF, gap_kernel='auto',
            on_collins_sampling='ignore')

    with pytest.raises(ValueError, match="Tracer"):
        jax.jit(bad)(env)
